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
            Dict: Node in parse tree.
        """
        node = self.root_node
        
        for depth in range(min(self.depth, len(tokens))):
            token = tokens[depth]
            
            if depth >= self.depth - 1:
                break
            
            if token not in node:
                node[token] = {}
            
            node = node[token]
        
        return node
    
    def _calculate_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """
        Calculate token sequence similarity using simplified matching.
        
        Args:
            tokens1: First token sequence.
            tokens2: Second token sequence.
            
        Returns:
            Similarity score [0, 1].
        """
        if len(tokens1) != len(tokens2):
            return 0
        
        match_count = sum(1 for t1, t2 in zip(tokens1, tokens2) if t1 == t2 or t2 == '<*>')
        return match_count / len(tokens1)
    
    def parse_log(self, log_line: str) -> Tuple[int, str]:
        """
        Parse a single log line and return template ID and template string.
        
        Args:
            log_line (str): A single log line.
            
        Returns:
            Tuple[int, str]: Template ID and template string.
        """
        tokens = self._tokenize(log_line)
        
        if not tokens:
            return None, None
        
        self.logs_processed += 1
        
        # Navigate to appropriate node
        node = self._get_node(tokens)
        
        # Find matching template
        matched_template_id = None
        max_similarity = 0
        
        if "templates" not in node:
            node["templates"] = {}
        
        for template_id, template_tokens in node["templates"].items():
            similarity = self._calculate_similarity(tokens, template_tokens)
            
            if similarity >= self.similarity_threshold and similarity > max_similarity:
                matched_template_id = template_id
                max_similarity = similarity
        
        # Create new template if no match
        if matched_template_id is None:
            matched_template_id = self.template_counter
            self.template_counter += 1
            node["templates"][matched_template_id] = tokens
            self.id_to_template[matched_template_id] = ' '.join(tokens)
            self.template_to_id[' '.join(tokens)] = matched_template_id
        else:
            # Generalise the template: positions that differ become <*>
            template_tokens = node["templates"][matched_template_id]
            merged = [t if t == s else '<*>' for t, s in zip(template_tokens, tokens)]
            if merged != template_tokens:
                node["templates"][matched_template_id] = merged
                self.id_to_template[matched_template_id] = ' '.join(merged)

        return matched_template_id, self.id_to_template[matched_template_id]
    
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
    
    def parse_from_list(self, log_lines: List[str]) -> Tuple[List[int], List[str]]:
        """
        Parse list of log lines (instead of reading from file).
        
        Args:
            log_lines: List of log message strings
            
        Returns:
            Tuple[List[int], List[str]]: Template IDs and template strings
        """
        template_ids = []
        template_strings = []
        
        for line_num, line in enumerate(log_lines):
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
