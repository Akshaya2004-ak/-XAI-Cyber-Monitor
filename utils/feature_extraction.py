# utils/feature_extraction.py
import re
import math
from urllib.parse import urlparse, parse_qs, unquote
from collections import Counter
import numpy as np

def entropy(s):
    """Calculate Shannon entropy of a string"""
    if len(s) == 0:
        return 0
    p = Counter(s)
    lns = float(len(s))
    return -sum(count / lns * math.log2(count / lns) for count in p.values() if count)

def extract_features(url):
    """
    Enhanced feature extraction for better ML model performance
    Extracts 15+ features for comprehensive analysis
    """
    try:
        # URL parsing
        parsed = urlparse(url)
        query = unquote(parsed.query) if parsed.query else ""
        path = unquote(parsed.path) if parsed.path else ""
        full_url = unquote(url)
        
        # Basic tokenization
        tokens = re.split(r'\W+', query.lower())
        path_tokens = re.split(r'\W+', path.lower())
        all_tokens = tokens + path_tokens
        
        # Token statistics
        token_lengths = [len(t) for t in tokens if t]
        token_count = len([t for t in tokens if t])
        token_length_sum = sum(token_lengths)
        avg_token_length = sum(token_lengths) / len(token_lengths) if token_lengths else 0
        max_token_length = max(token_lengths) if token_lengths else 0
        
        # Character statistics
        url_length = len(full_url)
        query_length = len(query)
        path_length = len(path)
        
        # Special character counts
        special_chars = len(re.findall(r'[<>"\';(){}[\]\\]', full_url))
        encoded_chars = len(re.findall(r'%[0-9A-Fa-f]{2}', url))
        numeric_chars = len(re.findall(r'\d', query))
        
        # SQL Injection patterns
        sql_keywords = ['select', 'union', 'insert', 'update', 'delete', 'drop', 'create', 'alter']
        sql_operators = ['or', 'and', 'not', 'like', 'in', 'exists']
        sql_functions = ['concat', 'substring', 'ascii', 'char', 'exec']
        
        sql_keyword_count = sum(1 for kw in sql_keywords if kw in query.lower())
        sql_operator_count = sum(1 for op in sql_operators if op in query.lower())
        sql_function_count = sum(1 for func in sql_functions if func in query.lower())
        
        # Advanced SQL patterns
        sql_comment_pattern = int(bool(re.search(r'(--|#|/\*)', query)))
        sql_quote_pattern = int(bool(re.search(r"'.*'|\".*\"", query)))
        sql_equals_pattern = int(bool(re.search(r"\b(or|and)\b.*=", query.lower())))
        sql_union_pattern = int(bool(re.search(r'union.*select', query.lower())))
        
        # XSS patterns
        xss_keywords = ['script', 'alert', 'prompt', 'confirm', 'eval', 'onclick', 'onload', 'onerror']
        xss_tags = ['<script>', '<iframe>', '<object>', '<embed>', '<form>', '<img>', '<svg>']
        
        xss_keyword_count = sum(1 for kw in xss_keywords if kw in full_url.lower())
        xss_tag_count = sum(1 for tag in xss_tags if tag in full_url.lower())
        
        # Advanced XSS patterns
        xss_event_handler = int(bool(re.search(r'on\w+\s*=', full_url.lower())))
        xss_javascript_protocol = int(bool(re.search(r'javascript:', full_url.lower())))
        xss_encoded_script = int(bool(re.search(r'%3[Cc]script', url)))
        xss_html_entities = int(bool(re.search(r'&[a-zA-Z]+;', full_url)))
        
        # Entropy calculations
        query_entropy = entropy(query)
        path_entropy = entropy(path)
        url_entropy = entropy(full_url)
        
        # Parameter analysis
        params = parse_qs(parsed.query)
        param_count = len(params)
        param_values = [v for values in params.values() for v in values]
        param_value_lengths = [len(v) for v in param_values]
        avg_param_length = sum(param_value_lengths) / len(param_value_lengths) if param_value_lengths else 0
        max_param_length = max(param_value_lengths) if param_value_lengths else 0
        
        # Suspicious parameter patterns
        suspicious_param_chars = sum(1 for v in param_values if any(c in v for c in '<>"\';()'))
        
        # Directory traversal patterns
        directory_traversal = int(bool(re.search(r'\.\./', full_url)))
        
        # File inclusion patterns  
        file_inclusion = int(bool(re.search(r'(file://|ftp://|data:)', full_url.lower())))
        
        # Command injection patterns
        command_injection = int(bool(re.search(r'[;&|`$()]', query)))
        
        # Feature vector (30 features total)
        features = np.array([
            # Basic statistics (5 features)
            token_count,
            token_length_sum,
            avg_token_length,
            max_token_length,
            url_length,
            
            # Character analysis (4 features)
            special_chars,
            encoded_chars,
            numeric_chars,
            query_length,
            
            # SQL injection features (8 features)
            sql_keyword_count,
            sql_operator_count,
            sql_function_count,
            sql_comment_pattern,
            sql_quote_pattern,
            sql_equals_pattern,
            sql_union_pattern,
            int(bool(re.search(r"'.*or.*'", query.lower()))),
            
            # XSS features (6 features)
            xss_keyword_count,
            xss_tag_count,
            xss_event_handler,
            xss_javascript_protocol,
            xss_encoded_script,
            xss_html_entities,
            
            # Entropy features (3 features)
            query_entropy,
            path_entropy,
            url_entropy,
            
            # Parameter analysis (3 features)
            param_count,
            avg_param_length,
            suspicious_param_chars,
            
            # Other attack patterns (3 features)
            directory_traversal,
            file_inclusion,
            command_injection
        ])
        
        return features
        
    except Exception as e:
        # Return zero vector if extraction fails
        return np.zeros(30)