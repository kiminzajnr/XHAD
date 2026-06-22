"""
Log Parser Module
Implements log parsing using Drain algorithm for template extraction.
"""

import re
import logging
from collections import defaultdict
from typing import Tuple, List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class DrainLogParser:
    """
    Log parser using Drain algorithm for template extraction.
    
    Attributes:
        depth (int): Depth of the parse tree for grouping.
        similarity_threshold (float): Threshold for token sequence similarity.
    """
    
    def __init__(self, depth: int = 4, similarity_threshold: float = 0.5):
        self.depth = depth
        self.similarity_threshold = similarity_threshold
        self.root_node = {}
        self.id_to_template = {}
        self.template_to_id = {}
        self.template_counter = 0
        self.logs_processed = 0
    
    def _tokenize(self, log_line: str) -> List[str]:
        """
        Tokenize a log line into individual tokens.
        
        Args:
            log_line (str): Raw log line.
            
        Returns:
            List[str]: List of tokens.
        """
        return log_line.strip().split()
    
    def _get_node(self, tokens: List[str]) -> Dict:
        """
        Navigate parse tree and return appropriate node.
        
        Args:
            tokens (List[str]): Log tokens.
            
        Returns:
            Dict: Node in the parse tree.
        """
        node = self.root_node
        
        for depth in range(min(self.depth, len(tokens))):
            token = tokens[depth]
            
            if token not in node:
                node[token] = {}
            
            node = node[token]
        
        return node
    
    def _calculate_similarity(self, template1: str, template2: str) -> float:
        """
        Calculate similarity between two template strings.
        
        Args:
            template1 (str): First template.
            template2 (str): Second template.
            
        Returns:
            float: Similarity score between 0 and 1.
        """
        tokens1 = template1.split()
        tokens2 = template2.split()
        
        if len(tokens1) != len(tokens2):
            return 0.0
        
        matching_tokens = sum(1 for t1, t2 in zip(tokens1, tokens2) if t1 == t2)
        similarity = matching_tokens / len(tokens1) if tokens1 else 0.0
        
        return similarity
    
    def parse_log(self, log_line: str) -> Tuple[int, str]:
        """
        Parse a single log line and extract template ID.
        
        Args:
            log_line (str): Raw log line.
            
        Returns:
            Tuple[int, str]: Template ID and template string.
        """
        tokens = self._tokenize(log_line)
        
        if not tokens:
            return None, None
        
        node = self._get_node(tokens)
        
        template_str = ' '.join(tokens)
        
        if template_str in self.template_to_id:
            template_id = self.template_to_id[template_str]
        else:
            self.template_counter += 1
            template_id = self.template_counter
            self.id_to_template[template_id] = template_str
            self.template_to_id[template_str] = template_id
        
        self.logs_processed += 1
        return template_id, template_str
    
    def parse_file(self, log_file: Path, max_logs: int = None) -> Tuple[List[int], List[str]]:
        """
        Parse entire log file.
        
        Args:
            log_file (Path): Path to log file.
            max_logs (int): Maximum number of logs to process. None = all.
            
        Returns:
            Tuple[List[int], List[str]]: Template IDs and template strings.
        """
        template_ids = []
        template_strings = []
        
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f):
                if max_logs and line_num >= max_logs:
                    break
                
                template_id, template_str = self.parse_log(line.strip())
                
                if template_id is not None:
                    template_ids.append(template_id)
                    template_strings.append(template_str)
                
                if (line_num + 1) % 500 == 0:
                    logger.info(f"Processed {line_num + 1} logs, {len(self.template_to_id)} unique templates")
        
        logger.info(f"Parsing complete: {self.logs_processed} logs, {len(self.template_to_id)} unique templates")
        return template_ids, template_strings
    
    def get_statistics(self) -> Dict:
        """
        Get parsing statistics.
        
        Returns:
            Dict: Statistics dictionary.
        """
        return {
            'logs_processed': self.logs_processed,
            'unique_templates': len(self.template_to_id),
            'avg_logs_per_template': self.logs_processed / len(self.template_to_id) if self.template_to_id else 0
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from config import Config
    
    parser = DrainLogParser(depth=Config.DRAIN_DEPTH, similarity_threshold=Config.DRAIN_SIMILARITY_THRESHOLD)
    
    print("Parsing HDFS sample logs...")
    hdfs_ids, hdfs_templates = parser.parse_file(Config.HDFS_LOG_PATH)
    print(f"HDFS: {len(hdfs_ids)} logs, {len(set(hdfs_ids))} unique templates")
    
    parser = DrainLogParser(depth=Config.DRAIN_DEPTH, similarity_threshold=Config.DRAIN_SIMILARITY_THRESHOLD)
    print("\nParsing BGL sample logs...")
    bgl_ids, bgl_templates = parser.parse_file(Config.BGL_LOG_PATH)
    print(f"BGL: {len(bgl_ids)} logs, {len(set(bgl_ids))} unique templates")