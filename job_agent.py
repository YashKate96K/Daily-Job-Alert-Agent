import os
import sys
import csv
import json
import ssl
import time
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from tavily import TavilyClient
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# ==============================================================================
# Top 10 Curated Tech & Internship Platforms (India + Global Remote)
# Restricts searches to high-signal job boards, eliminating courses & blog spam.
# ==============================================================================
TOP_JOB_DOMAINS = [
    "linkedin.com",          # #1 professional network & active recruiter postings
    "wellfound.com",         # Top hub for high-growth tech & AI startups
    "ycombinator.com",       # YC Work at a Startup (cutting-edge AI/founder roles)
    "internshala.com",       # #1 dedicated student internship platform in India
    "instahyre.com",         # Premium tech hiring (Flipkart, Swiggy, Uber, top startups)
    "cuvette.tech",          # Platform tailored for student tech/SDE/ML internships
    "unstop.com",            # Popular in Indian colleges for tech hiring challenges
    "indeed.co.in",          # Broad recruiter reach across Indian tech hubs
    "boards.greenhouse.io",  # Direct company ATS portal (clean, no middleman spam)
    "jobs.lever.co",         # Direct company ATS portal (clean, verified jobs)
]


def safe_print(text: str) -> None:
    """Utility to print text safely without crashing on non-UTF8 / Windows console charmaps."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding))


def retry_with_backoff(operation, retries: int = 2, delay: float = 2.0, description: str = "API operation"):
    """
    Stage 5: Retry Logic with Exponential Backoff
    
    ROBUSTNESS & ERROR RECOVERY:
    External web APIs (Tavily search, Groq LLM inference) can experience transient network
    disconnects, rate limits (HTTP 429), or temporary server congestion (HTTP 503).
    
    This function wraps any callable operation, retrying up to `retries` times with an
    increasing backoff delay before allowing an exception to raise.
    
    Args:
        operation: Callable lambda/function to execute.
        retries: Number of retry attempts after initial failure (default: 2 retries = 3 attempts total).
        delay: Initial sleep duration in seconds (doubles each attempt).
        description: Informative label for console logging.
    """
    current_delay = delay
    for attempt in range(retries + 1):
        try:
            return operation()
        except Exception as e:
            if attempt < retries:
                print(f"  [Retry Warning] {description} failed ({e}). Retrying in {current_delay}s... (Attempt {attempt + 1}/{retries})")
                time.sleep(current_delay)
                current_delay *= 2  # Exponential backoff: 2s -> 4s
            else:
                print(f"  [Graceful Failure] {description} failed after {retries} retries: {e}")
                raise e


def search_jobs(query: str, max_results: int = 5, domains: list[str] = None) -> list[dict]:
    """
    Stage 1 & 5: Search Function with Retry Logic & Domain Targeting
    Calls Tavily Search API with automatic retries on network/rate-limit errors.
    
    Args:
        query: The search string.
        max_results: Max number of results to fetch.
        domains: Optional list of domains to restrict search to (e.g. TOP_JOB_DOMAINS).
        
    Returns:
        List of dictionaries with keys: 'title', 'link', and 'snippet'.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY not found in environment variables. "
            "Please create a .env file based on .env.example."
        )

    client = TavilyClient(api_key=api_key)

    search_kwargs = {
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
    }
    if domains:
        search_kwargs["include_domains"] = domains

    try:
        # Wrap the network call in retry logic
        response = retry_with_backoff(
            lambda: client.search(**search_kwargs),
            retries=2,
            delay=2.0,
            description=f"Tavily Search ('{query}')"
        )
    except Exception as e:
        # Give up gracefully and return empty list so the agent loop can continue
        print(f"  [Search Skipped] Giving up on search query '{query}' due to error: {e}")
        return []

    raw_results = []
    for item in response.get("results", []):
        raw_results.append({
            "title": item.get("title", "No Title"),
            "link": item.get("url", ""),
            "snippet": item.get("content", ""),
        })

    return raw_results


def filter_jobs_with_llm(raw_results: list[dict]) -> list[dict]:
    """
    Stage 2 & 5: LLM Filtering with Retry Logic
    Uses Groq's LLM to evaluate raw search results against target criteria with retry protection:
      1. AI/ML domain (Machine Learning, Deep Learning, Data Science, Generative AI, NLP, CV).
      2. Internship or Entry-level.
      3. India-based or Remote.
      4. Recent/active job postings.
    """
    if not raw_results:
        return []

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY not found in environment variables. "
            "Please add GROQ_API_KEY to your .env file."
        )

    client = Groq(api_key=groq_api_key)
    model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    system_prompt = (
        "You are an expert career assistant evaluating raw job search results for tech internships.\n"
        "Evaluate each item against these criteria:\n"
        "1. Eligible Domains: AI, Machine Learning (ML), Deep Learning (DL), Generative AI, NLP, "
        "Computer Vision, Software Engineering (SDE/SWE/Backend/Fullstack), Data Science, or Data Engineering.\n"
        "2. Level: Internship, Trainee, or Entry-level (reject Senior, Staff, Lead, Manager roles).\n"
        "3. Location: India-based or open to Remote.\n"
        "4. Authenticity: Must be an actual job/internship posting (reject courses, tutorials, syllabus ads, or spam).\n\n"
        "Return a valid JSON object with a single key 'results' containing an array of objects.\n"
        "Each object must have exactly these keys:\n"
        "- title: Job title (string)\n"
        "- company: Company name if identifiable, otherwise 'Unknown' (string)\n"
        "- domain: Classified domain ('AI/ML', 'Deep Learning', 'Software Engineering (SDE)', 'Data Science', or 'Other Tech') (string)\n"
        "- link: The original job link (string)\n"
        "- relevant: true or false (boolean)\n"
        "- reason: Short one-sentence explanation of why it is or isn't relevant (string)\n"
    )

    user_prompt = f"Raw Search Results:\n{json.dumps(raw_results, indent=2)}"

    try:
        # Wrap LLM inference in retry logic
        chat_completion = retry_with_backoff(
            lambda: client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            ),
            retries=2,
            delay=2.0,
            description="Groq LLM Filter"
        )
        response_text = chat_completion.choices[0].message.content
        parsed = json.loads(response_text)

        if isinstance(parsed, dict):
            return parsed.get("results") or parsed.get("jobs") or []
        elif isinstance(parsed, list):
            return parsed
        return []
    except Exception as e:
        # Give up gracefully and return empty list
        print(f"  [Filter Skipped] Giving up on LLM filter due to error: {e}")
        return []


def generate_alternative_query(previous_queries: list[str], reasons_for_rejection: list[str]) -> str:
    """
    Stage 3 & 5: Query Re-formulation with Retry Logic and Fallback
    """
    fallback_queries = [
        "Machine Learning intern hiring freshers India 2025",
        "Junior Data Scientist AI intern remote India apply",
        "Deep Learning AI internship Bangalore Gurgaon India"
    ]

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        for fq in fallback_queries:
            if fq not in previous_queries:
                return fq
        return f"{previous_queries[-1]} hiring now"

    client = Groq(api_key=groq_api_key)
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    prompt = (
        "You are an autonomous job search agent. Your previous web search queries did not yield "
        "enough relevant entry-level AI/ML internships in India.\n\n"
        f"Previous queries tried: {previous_queries}\n"
        f"Sample reasons why previous results were rejected: {reasons_for_rejection[:3]}\n\n"
        "Generate ONE new, refined, and distinct web search query to find actual active AI/ML internship postings.\n"
        "Tips: use specific phrases like 'apply', 'hiring', 'intern', 'careers', and target India/remote.\n"
        "Output ONLY the new query string, without quotes or additional text."
    )

    try:
        chat_completion = retry_with_backoff(
            lambda: client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            ),
            retries=2,
            delay=2.0,
            description="Groq LLM Re-planning"
        )
        return chat_completion.choices[0].message.content.strip().strip('"').strip("'")
    except Exception as e:
        print(f"  [Re-plan Fallback] LLM re-planning call failed ({e}). Using rule-based fallback query.")
        for fq in fallback_queries:
            if fq not in previous_queries:
                return fq
        return f"{previous_queries[-1]} hiring now"


def format_email_body(relevant_jobs: list[dict]) -> tuple[str, str]:
    """
    Stage 4 Helper: Format both Plain Text and HTML versions of the email alert.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Plain text version (clean ASCII for reliable console/email client rendering)
    if not relevant_jobs:
        plain_text = f"AI/ML Job Alert Agent ({now_str})\n\nNo new relevant internship postings found today."
    else:
        plain_text = f"[ALERT] AI/ML Internship Opportunities ({now_str})\nFound {len(relevant_jobs)} relevant openings:\n\n"
        for i, job in enumerate(relevant_jobs, 1):
            plain_text += f"{i}. {job.get('title')}\n"
            plain_text += f"   Company: {job.get('company', 'Unknown')}\n"
            plain_text += f"   Why it matches: {job.get('reason', '')}\n"
            plain_text += f"   Link: {job.get('link')}\n\n"
        plain_text += "Automated daily alert generated by your Job Alert Agent."

    # Responsive HTML version
    if not relevant_jobs:
        job_cards_html = "<p style='color: #64748b;'>No new relevant internship postings matching your criteria were found today.</p>"
    else:
        cards = []
        for i, job in enumerate(relevant_jobs, 1):
            cards.append(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <div style="margin-bottom: 6px;">
                    <h3 style="margin: 0; font-size: 16px; color: #1e293b;">
                        <a href="{job.get('link')}" target="_blank" style="color: #2563eb; text-decoration: none; font-weight: 600;">
                            {i}. {job.get('title')}
                        </a>
                    </h3>
                </div>
                <div style="margin-bottom: 10px;">
                    <span style="background: #f1f5f9; color: #475569; font-size: 12px; font-weight: 600; padding: 3px 8px; border-radius: 4px;">
                        Company: {job.get('company', 'Unknown')}
                    </span>
                </div>
                <p style="margin: 8px 0 14px 0; color: #334155; font-size: 14px; line-height: 1.5;">
                    <strong>Why it matches:</strong> {job.get('reason', 'N/A')}
                </p>
                <a href="{job.get('link')}" target="_blank" style="display: inline-block; background: #2563eb; color: #ffffff; padding: 7px 16px; border-radius: 6px; text-decoration: none; font-size: 13px; font-weight: 500;">
                    View & Apply &rarr;
                </a>
            </div>
            """)
        job_cards_html = "\n".join(cards)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"></head>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f8fafc; margin: 0; padding: 24px; color: #0f172a;">
        <div style="max-width: 600px; margin: 0 auto; background: #ffffff; border-radius: 12px; padding: 28px; border: 1px solid #e2e8f0;">
            <div style="border-bottom: 2px solid #3b82f6; padding-bottom: 14px; margin-bottom: 20px;">
                <h1 style="margin: 0; font-size: 22px; color: #1e293b;">AI/ML Internship Alert</h1>
                <p style="margin: 4px 0 0 0; color: #64748b; font-size: 13px;">Daily scan &bull; {now_str} &bull; {len(relevant_jobs)} opportunities found</p>
            </div>
            {job_cards_html}
            <div style="margin-top: 24px; padding-top: 14px; border-top: 1px solid #e2e8f0; font-size: 12px; color: #94a3b8; text-align: center;">
                Generated by Job Alert Agent &bull; Autonomous AI Internship Hunter
            </div>
        </div>
    </body>
    </html>
    """
    return plain_text, html_content


def send_email_alert(relevant_jobs: list[dict], recipient: str = None) -> bool:
    """
    Stage 4: Email Delivery
    Sends a formatted HTML & plain-text email with the filtered relevant jobs using SMTP (Gmail App Password).
    
    If email environment variables are not configured, it gracefully displays an email
    preview in the console without crashing.
    """
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "465"))
    sender_email = os.getenv("EMAIL_SENDER")
    sender_password = os.getenv("EMAIL_PASSWORD")
    target_recipient = recipient or os.getenv("EMAIL_RECIPIENT")

    plain_text, html_content = format_email_body(relevant_jobs)

    # Check if credentials are provided in .env
    if not sender_email or not sender_password or not target_recipient:
        print("\n[Stage 4 - Email Notice]")
        print("EMAIL_SENDER, EMAIL_PASSWORD, or EMAIL_RECIPIENT not set in .env.")
        print("Displaying formatted email preview below:\n")
        print("=" * 60)
        print(f"To: {target_recipient or '(Recipient not specified)'}")
        print(f"Subject: [ALERT] AI/ML Internship Alert ({len(relevant_jobs)} openings found)")
        print("-" * 60)
        safe_print(plain_text)
        print("=" * 60)
        return False

    # Create MIMEMultipart message with both plain and HTML alternatives
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"AI/ML Internship Alert: {len(relevant_jobs)} New Opportunities Found"
    msg["From"] = sender_email
    msg["To"] = target_recipient

    msg.attach(MIMEText(plain_text, "plain", "utf-8"))
    msg.attach(MIMEText(html_content, "html", "utf-8"))

    try:
        if smtp_port == 465:
            # SSL connection
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, target_recipient, msg.as_string())
        else:
            # STARTTLS connection (e.g., port 587)
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, target_recipient, msg.as_string())

        print(f"\n[Stage 4 - Success] Email alert sent successfully to {target_recipient}!")
        return True
    except Exception as e:
        print(f"\n[Stage 4 - Error] Failed to send email via SMTP: {e}")
        return False


def log_run_to_csv(
    queries_used: list[str],
    total_found: int,
    relevant_count: int,
    status: str,
    csv_path: str = "agent_runs.csv",
    error_message: str = ""
) -> None:
    """
    Stage 6: Persistent Logging to CSV
    
    OBSERVABILITY & AUDIT TRAIL:
    Autonomous background agents require transparent tracking. This function appends
    an audit row after every run (timestamp, queries attempted, counts, status).
    """
    file_exists = os.path.exists(csv_path)
    headers = [
        "timestamp",
        "queries_used",
        "total_results_found",
        "relevant_results_count",
        "status",
        "error_message"
    ]

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    joined_queries = " | ".join(queries_used)

    with open(csv_path, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "timestamp": timestamp_str,
            "queries_used": joined_queries,
            "total_results_found": total_found,
            "relevant_results_count": relevant_count,
            "status": status,
            "error_message": error_message,
        })

    print(f"[Stage 6 - Logging] Run metrics successfully recorded to '{csv_path}'.")


def run_job_agent(
    initial_query: str = "AI ML Deep Learning Software Engineer intern hiring India 2025",
    min_relevant: int = 2,
    max_searches: int = 3,
    use_top_domains: bool = True,
    send_email: bool = True,
    csv_log_path: str = "agent_runs.csv"
) -> dict:
    """
    End-to-End Autonomous Multi-Domain Agent Execution:
      1. Search Tavily across Top 10 Job Portals (with retry logic)
      2. Filter relevance & categorize domains via Groq LLM (with retry logic)
      3. Agentic re-planning loop across ML/DL/SDE tracks (hard budget cap)
      4. Email delivery with categorized badges (HTML & plain-text via SMTP)
      5. CSV execution logging (auditability)
    """
    target_domains = TOP_JOB_DOMAINS if use_top_domains else None
    queries_used = [initial_query]
    all_evaluated = []
    seen_links = set()
    search_count = 0
    current_query = initial_query
    status = "SUCCESS"
    error_msg = ""

    domain_info = f"Top {len(target_domains)} Curated Job Portals" if target_domains else "Open Web"
    print(f"=== Multi-Domain Job Alert Agent Launching ===")
    print(f"Target Search Scope: {domain_info}")
    print(f"Domains tracked: AI/ML, Deep Learning, Software Engineering (SDE), Data Science")

    try:
        while search_count < max_searches:
            search_count += 1
            print(f"\n[Iteration {search_count}/{max_searches}] Searching with query: '{current_query}'")

            # 1. Search Tavily across Top 10 Job Portals (with retry logic)
            raw_results = search_jobs(current_query, max_results=5, domains=target_domains)
            print(f"  -> Retrieved {len(raw_results)} job listings from {'target portals' if target_domains else 'web'}.")

            # Deduplicate results across iterations by URL
            new_results = [r for r in raw_results if r["link"] not in seen_links]
            for r in new_results:
                seen_links.add(r["link"])

            if not new_results:
                print("  -> All retrieved results were already seen in previous searches.")

            # 2. Filter with LLM (with retry logic)
            print("  -> Evaluating relevance and categorizing domains with LLM...")
            filtered_batch = filter_jobs_with_llm(new_results)
            all_evaluated.extend(filtered_batch)

            # Count total relevant jobs collected so far
            relevant_jobs = [job for job in all_evaluated if job.get("relevant") is True]
            print(f"  -> Relevant jobs found so far: {len(relevant_jobs)} (Target: {min_relevant})")

            # 3. Reflection / Re-planning Decision
            if len(relevant_jobs) >= min_relevant:
                print(f"[Success] Found sufficient relevant jobs ({len(relevant_jobs)} >= {min_relevant}). Stopping loop.")
                break

            # If budget exhausted, stop
            if search_count >= max_searches:
                print(f"[Budget Cap Reached] Reached maximum allowed searches ({max_searches}). Halting search loop.")
                status = "BUDGET_EXHAUSTED"
                break

            # Re-plan: Gather reasons from non-relevant jobs to guide the next query (with retry logic)
            rejections = [job.get("reason", "") for job in filtered_batch if not job.get("relevant")]
            print("  -> Insufficient relevant results. Re-planning search query using LLM...")
            current_query = generate_alternative_query(queries_used, rejections)
            queries_used.append(current_query)

        final_relevant = [job for job in all_evaluated if job.get("relevant") is True]

        # Stage 4: Deliver email
        if send_email:
            send_email_alert(final_relevant)

    except Exception as exc:
        status = "FAILURE"
        error_msg = str(exc)
        print(f"\n[Agent Error] Run encountered fatal exception: {exc}")
        final_relevant = [job for job in all_evaluated if job.get("relevant") is True]
    finally:
        # Stage 6: Always log run outcome to CSV
        log_run_to_csv(
            queries_used=queries_used,
            total_found=len(all_evaluated),
            relevant_count=len(final_relevant),
            status=status,
            csv_path=csv_log_path,
            error_message=error_msg
        )

    return {
        "queries_used": queries_used,
        "search_count": search_count,
        "total_evaluated": len(all_evaluated),
        "relevant_jobs": final_relevant,
        "all_jobs": all_evaluated,
        "status": status,
    }


if __name__ == "__main__":
    tavily_key = os.getenv("TAVILY_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    if tavily_key and groq_key:
        results = run_job_agent(
            initial_query="AI ML Deep Learning Software Engineer intern hiring India 2025",
            min_relevant=2,
            max_searches=3,
            use_top_domains=True,
            send_email=True
        )
        print("\n=== Agent Run Finished ===")
        print(f"Status: {results['status']}")
        print(f"Total relevant opportunities: {len(results['relevant_jobs'])}")
        for job in results['relevant_jobs']:
            print(f" - [{job.get('domain', 'Tech')}] {job.get('title')} @ {job.get('company')} ({job.get('link')})")
    else:
        print("[Notice] Please configure TAVILY_API_KEY and GROQ_API_KEY in your .env file.")
