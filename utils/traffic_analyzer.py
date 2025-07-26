# utils/traffic_analyzer.py
import re
from urllib.parse import urlparse
import numpy as np

class TrafficAnalyzer:
    """
    AI-Powered Traffic Classification System
    Implements Deliverable 1: Traffic Classification based on behavior and patterns
    """
    
    def __init__(self):
        self.traffic_patterns = {
            'API': [r'/api/', r'/rest/', r'/v\d+/', r'\.json', r'\.xml'],
            'Media': [r'\.(jpg|jpeg|png|gif|mp4|mp3|avi|mov|webm)', r'/media/', r'/images/', r'/videos/'],
            'Email': [r'/mail/', r'/email/', r'/webmail/', r'/inbox/', r'/compose/'],
            'Social': [r'/social/', r'/feed/', r'/post/', r'/share/', r'/like/', r'/comment/'],
            'Ecommerce': [r'/shop/', r'/cart/', r'/checkout/', r'/product/', r'/buy/', r'/payment/'],
            'File_Transfer': [r'/download/', r'/upload/', r'/ftp/', r'\.(pdf|doc|zip|exe|dmg)'],
            'Streaming': [r'/stream/', r'/video/', r'/live/', r'/watch/', r'/play/'],
            'Authentication': [r'/login/', r'/auth/', r'/signin/', r'/register/', r'/oauth/'],
            'Database': [r'/db/', r'/database/', r'/query/', r'/admin/'],
            'CDN': [r'cdn\.', r'static\.', r'assets\.', r'/static/', r'/assets/'],
            'Search': [r'/search/', r'/find/', r'\?q=', r'\?query='],
            'Gaming': [r'/game/', r'/play/', r'/score/', r'/level/']
        }
    
    def classify_traffic(self, url):
        """
        Classify network traffic based on URL patterns
        Returns traffic type for APP ID detection
        """
        url_lower = url.lower()
        parsed = urlparse(url_lower)
        full_path = f"{parsed.path}?{parsed.query}" if parsed.query else parsed.path
        
        # Check for suspicious patterns first
        if self._is_suspicious(url):
            return "Suspicious"
        
        # Score each traffic type
        scores = {}
        for traffic_type, patterns in self.traffic_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, url_lower):
                    score += 1
            scores[traffic_type] = score
        
        # Return the highest scoring type
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        else:
            return "General Web"
    
    def _is_suspicious(self, url):
        """
        Check for suspicious patterns that might indicate attacks
        """
        suspicious_patterns = [
            r'<script',
            r'javascript:',
            r'alert\(',
            r'onerror=',
            r'onload=',
            r'union.*select',
            r'drop.*table',
            r'1=1',
            r"'.*or.*'",
            r'--|#',
            r'\.\./',
            r'%3C.*%3E',  # URL encoded < >
            r'%27',       # URL encoded '
            r'%22',       # URL encoded "
        ]
        
        url_lower = url.lower()
        for pattern in suspicious_patterns:
            if re.search(pattern, url_lower):
                return True
        return False
    
    def get_traffic_stats(self, urls):
        """
        Analyze traffic patterns for a list of URLs
        Returns comprehensive traffic statistics
        """
        traffic_types = [self.classify_traffic(url) for url in urls]
        
        # Basic stats
        unique_types = list(set(traffic_types))
        type_counts = {t: traffic_types.count(t) for t in unique_types}
        
        # Suspicious traffic percentage
        suspicious_count = traffic_types.count("Suspicious")
        suspicious_percentage = (suspicious_count / len(urls)) * 100 if urls else 0
        
        return {
            'total_requests': len(urls),
            'unique_traffic_types': len(unique_types),
            'traffic_distribution': type_counts,
            'suspicious_traffic_count': suspicious_count,
            'suspicious_traffic_percentage': suspicious_percentage,
            'dominant_traffic_type': max(type_counts, key=type_counts.get) if type_counts else None
        }
    
    def detect_anomalies(self, urls, threshold=0.1):
        """
        Detect traffic anomalies based on unusual patterns
        Implements part of Deliverable 2: Anomaly Identification
        """
        anomalies = []
        
        for url in urls:
            anomaly_score = 0
            reasons = []
            
            # Check for unusual characters
            unusual_chars = len(re.findall(r'[<>"\';(){}[\]\\]', url))
            if unusual_chars > 5:
                anomaly_score += 0.3
                reasons.append(f"High unusual character count: {unusual_chars}")
            
            # Check for excessive URL encoding
            encoded_chars = len(re.findall(r'%[0-9A-Fa-f]{2}', url))
            if encoded_chars > 10:
                anomaly_score += 0.2
                reasons.append(f"Excessive URL encoding: {encoded_chars}")
            
            # Check for very long URLs
            if len(url) > 500:
                anomaly_score += 0.2
                reasons.append(f"Very long URL: {len(url)} characters")
            
            # Check for suspicious keywords
            suspicious_keywords = ['script', 'alert', 'drop', 'union', 'select', 'exec']
            found_keywords = [kw for kw in suspicious_keywords if kw in url.lower()]
            if found_keywords:
                anomaly_score += 0.4
                reasons.append(f"Suspicious keywords: {found_keywords}")
            
            # Check for multiple parameter injection attempts
            param_injections = len(re.findall(r'[=&].*[<>\'";]', url))
            if param_injections > 3:
                anomaly_score += 0.3
                reasons.append(f"Multiple parameter injections: {param_injections}")
            
            if anomaly_score >= threshold:
                anomalies.append({
                    'url': url,
                    'anomaly_score': anomaly_score,
                    'reasons': reasons,
                    'severity': 'High' if anomaly_score > 0.7 else 'Medium' if anomaly_score > 0.4 else 'Low'
                })
        
        return anomalies
    
    def generate_traffic_report(self, urls):
        """
        Generate comprehensive traffic analysis report
        Combines both deliverables into a single report
        """
        stats = self.get_traffic_stats(urls)
        anomalies = self.detect_anomalies(urls)
        
        report = {
            'summary': {
                'total_analyzed': len(urls),
                'traffic_types_detected': stats['unique_traffic_types'],
                'anomalies_detected': len(anomalies),
                'threat_level': 'High' if stats['suspicious_traffic_percentage'] > 20 else 
                              'Medium' if stats['suspicious_traffic_percentage'] > 5 else 'Low'
            },
            'traffic_classification': stats,
            'anomaly_detection': {
                'total_anomalies': len(anomalies),
                'high_severity': len([a for a in anomalies if a['severity'] == 'High']),
                'medium_severity': len([a for a in anomalies if a['severity'] == 'Medium']),
                'low_severity': len([a for a in anomalies if a['severity'] == 'Low']),
                'anomalies': anomalies
            },
            'recommendations': self._generate_recommendations(stats, anomalies)
        }
        
        return report
    
    def _generate_recommendations(self, stats, anomalies):
        """
        Generate security recommendations based on analysis
        """
        recommendations = []
        
        if stats['suspicious_traffic_percentage'] > 10:
            recommendations.append("High suspicious traffic detected. Implement additional monitoring.")
        
        if len(anomalies) > 0:
            high_severity = len([a for a in anomalies if a['severity'] == 'High'])
            if high_severity > 0:
                recommendations.append(f"Critical: {high_severity} high-severity anomalies detected. Immediate investigation required.")
        
        if 'API' in stats['traffic_distribution'] and stats['traffic_distribution']['API'] > len(stats['traffic_distribution']) * 0.3:
            recommendations.append("High API traffic detected. Ensure API security measures are in place.")
        
        if stats['suspicious_traffic_count'] == 0:
            recommendations.append("No immediate threats detected. Continue monitoring.")
        
        return recommendations