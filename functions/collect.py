import time
from github import Github, Auth
from github.GithubException import RateLimitExceededException

def check_remaining_queries(token):
    """
    Check the number of remaining API queries for the authenticated user.
    
    Args:
        github_auth (str): The GitHub authentication token.
        
    Returns:
        int: The number of remaining queries.
    """
    auth = Auth.Token(token)
    g = Github(auth=auth)
    rate_limit = g.get_rate_limit()
    return rate_limit.core.remaining if rate_limit else 0

def scrape_github(token, max_repos=100, min_stars=100):
    """
    Scrape GitHub for popular Python repositories (regardless of dependency file)
    and return the raw repository objects. If rate limited, stop and return
    whatever has been collected so far.
    
    Returns:
        dict: A dictionary mapping repository full names to their GitHub repository objects.
    """
    # Replace with your own GitHub personal access token.
    auth = Auth.Token(token)
    g = Github(auth=auth)
    
    # Use a single query that gets Python repositories above the min_stars threshold.
    query = f'language:Python stars:>={min_stars}'
    
    raw_repos = {}
    print("Gathering repository data from GitHub...")

    try:
        repositories = g.search_repositories(query, sort="stars", order="desc")
    except Exception as e:
        print("Error querying GitHub:", e)
        return raw_repos

    try:
        # Iterate over the repository objects until we collect up to max_repos.
        for repo in repositories:
            if len(raw_repos) >= max_repos:
                break
            if repo.full_name in raw_repos:
                continue
            raw_repos[repo.full_name] = repo
            print(f"Retrieved: {repo.full_name}")
            # Sleep a bit to be kind to the API.
            time.sleep(0.1)
    except RateLimitExceededException:
        print("Rate limit reached. Returning collected repositories.")
        return raw_repos
    except Exception as e:
        # Optionally, check if the exception is rate-limit related.
        if "rate limit" in str(e).lower():
            print("Rate limit reached (detected by exception message). Returning collected repositories.")
            return raw_repos
        else:
            print("An unexpected error occurred:", e)

    print(f"Collected {len(raw_repos)} repositories.")
    return raw_repos


def extract_dependencies(raw_repos):
    """
    Given a dictionary of raw GitHub repository objects, extract and deduplicate
    dependencies from each repo by looking for common dependency files.
    
    This version parallelizes the process using ThreadPoolExecutor.
    
    Returns:
        dict: A dictionary mapping repository full names to a list of normalized dependencies.
    """
    import re, ast, time
    from configparser import ConfigParser
    import toml
    import concurrent.futures

    # Define helper functions for each file type.
    def extract_packages_from_requirements(text):
        packages = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            line = re.split(r'\s+#', line)[0]
            pkg = re.split(r'[<=>]', line)[0].strip()
            if pkg:
                packages.append(pkg.lower())
        return packages

    def extract_packages_from_setup_py(text):
        pattern = re.compile(r'install_requires\s*=\s*(\[[^\]]*\])', re.MULTILINE | re.DOTALL)
        match = pattern.search(text)
        if match:
            list_str = match.group(1)
            try:
                deps = ast.literal_eval(list_str)
                packages = []
                for dep in deps:
                    dep_clean = re.split(r'[<=>]', dep)[0].strip().lower()
                    if dep_clean:
                        packages.append(dep_clean)
                return packages
            except Exception:
                return []
        return []

    def extract_packages_from_setup_cfg(text):
        config = ConfigParser()
        config.read_string(text)
        if config.has_option('options', 'install_requires'):
            deps_str = config.get('options', 'install_requires')
            packages = []
            for line in deps_str.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    pkg = re.split(r'[<=>]', line)[0].strip()
                    if pkg:
                        packages.append(pkg.lower())
            return packages
        return []

    def extract_packages_from_pyproject(text):
        try:
            data = toml.loads(text)
            if 'project' in data and 'dependencies' in data['project']:
                deps = data['project']['dependencies']
                packages = []
                for dep in deps:
                    pkg = re.split(r'[<=>]', dep)[0].strip()
                    if pkg:
                        packages.append(pkg.lower())
                return packages
            elif 'tool' in data and 'poetry' in data['tool']:
                poetry_deps = data['tool']['poetry'].get('dependencies', {})
                packages = [pkg.lower() for pkg in poetry_deps if pkg.lower() != 'python']
                return packages
        except Exception:
            return []
        return []
    
    def extract_from_file(file_name, file_text):
        if file_name.endswith("requirements.txt"):
            return extract_packages_from_requirements(file_text)
        elif file_name == "setup.py":
            return extract_packages_from_setup_py(file_text)
        elif file_name == "setup.cfg":
            return extract_packages_from_setup_cfg(file_text)
        elif file_name.endswith("pyproject.toml"):
            return extract_packages_from_pyproject(file_text)
        else:
            return extract_packages_from_requirements(file_text)
    
    dependency_files = ["requirements.txt", "setup.py", "setup.cfg", "pyproject.toml"]
    repos_dependencies = {}

    # Define a function to process a single repository.
    def process_repo(repo):
        aggregated_packages = []
        for file_name in dependency_files:
            try:
                content_file = repo.get_contents(file_name)
                file_text = content_file.decoded_content.decode("utf-8", errors="replace")
                packages = extract_from_file(file_name, file_text)
                aggregated_packages.extend(packages)
            except Exception:
                continue
        aggregated_packages = list(set(aggregated_packages))
        if aggregated_packages:
            # Optionally, sleep a bit to ease rate limits.
            time.sleep(0.1)
            return repo.full_name, aggregated_packages
        return None

    print("Extracting dependencies from repositories (parallelized)...")
    # Parallelize processing using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Submit a task for each repository
        future_to_repo = {executor.submit(process_repo, repo): repo for repo in raw_repos.values()}
        for future in concurrent.futures.as_completed(future_to_repo):
            result = future.result()
            if result:
                repo_name, aggregated_packages = result
                repos_dependencies[repo_name] = aggregated_packages
                print(f"Processed: {repo_name} with {len(aggregated_packages)} package(s)")

    print(f"Extracted dependencies from {len(repos_dependencies)} repositories.")
    return repos_dependencies
