import json
import os
import re
from typing import Any, Iterable
from openai import OpenAI
import jsonlines
from datetime import datetime
import urllib.request as libreq
import feedparser
import io

ARXIV_ABSTRACTS_PATH_TEMPLATE = 'arxiv_abstracts/{}.json'

MAX_NUM_USERS = 8


def fetch_arxiv_abstract(catogory: str, date: datetime, batch_size: int = 100, sortBy: str = 'submittedDate', sortOrder: str = 'descending', MAX_RESULT_LIMIT=1000) -> list:
    daily_entries = []

    reached_end = False
    for i in range(0, MAX_RESULT_LIMIT, batch_size):
        start = i
        max_results = min(batch_size, MAX_RESULT_LIMIT - i)
        url = f'http://export.arxiv.org/api/query?search_query=cat:{catogory}&start={start}&max_results={max_results}&sortBy={sortBy}&sortOrder={sortOrder}'
        response = libreq.urlopen(url).read()
        feed = feedparser.parse(response).entries

        for entry in feed:
            published = entry.published_parsed
            if published.tm_year == date.year and published.tm_mon == date.month and published.tm_mday == date.day:
                daily_entries.append(entry)
            elif (published.tm_year, published.tm_mon, published.tm_mday) < (date.year, date.month, date.day):
                reached_end = True
                break
        if reached_end:
            break

    if not reached_end:
        assert len(daily_entries) == MAX_RESULT_LIMIT, f'Number of entries does not match MAX_RESULT_LIMIT={MAX_RESULT_LIMIT}, unexpected error.'
        print(f'Warning: Number of entries is equal to MAX_RESULT_LIMIT={MAX_RESULT_LIMIT}. There might be more entries to fetch.')

    return daily_entries


def llm_summary_submit(entries: Iterable[Any], api_key: str, users: list):
    SYSTEM_PROMPT_KEYWORDS = '''You will receive the title and abstract of a research paper in the field of Machine Learning (Computer Science) along with a description of several reseachers' research interests. Your tasks are:

1. Identify the paper’s precise subfield using up to 5 keywords (e.g., Reinforcement Learning from Human Feedback, Neural Architecture Search).
2. Summarize the paper in 2-3 concise and informative sentences.
3. Rate the relevance of the paper to the each researcher's research interests on a scale of 1 to 5, with 5 being most relevant, and provide a brief justification.

Follow this exact format for your response:

    keywords: <keyword1>, <keyword2>, ...
    summary: <Your summary here.>
    relevance to researcher 1: <1-5>, <Your explanation here.>
    relevance to researcher 2: <1-5>, <Your explanation here.>
    ...

Do not add extra characters or spaces.
    '''

    USER_PROMPT_KEYWORDS = '''Here's the title and abstract of a research paper: 
    Title: {title}

    Abstract: {abstract}
    
    User's research interests: 
    {interests}
    '''

    TMP_FILE = 'tmp/llm_summary_prompt.jsonl'

    entry_ids = []

    client = OpenAI(api_key=api_key)
    with jsonlines.open(TMP_FILE, mode='w') as writer:
        for entry in entries:
            id = entry["id"].removeprefix('http://arxiv.org/abs/')
            title = entry["title"]
            abstract = entry["summary"]
            interests = []
            for i, user in enumerate(users):
                interests.append(f"Researcher {i+1}: {user['interests']}")
            prompt = USER_PROMPT_KEYWORDS.format(title=title, abstract=abstract, interests="\n".join(interests))
            writer.write({"custom_id": id, "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-mini", "messages": [{"role": "system", "content": SYSTEM_PROMPT_KEYWORDS},{"role": "user", "content": prompt}],"max_tokens": 2048}})
            entry_ids.append(id)
    
    batch_input_file = client.files.create(
        file=open(TMP_FILE, "rb"),
        purpose="batch"
    )

    batch_input_file_id = batch_input_file.id

    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description": "arxiv_abstract " + datetime.now().strftime("%Y%m%d%H%M%S")
        }
    )

    return batch, entry_ids

def llm_summary_fetch(api_key: str, batch_id: str):
    client = OpenAI(api_key=api_key)
    batch = client.batches.retrieve(batch_id=batch_id)
    if batch.status == 'completed':
        file_response = client.files.content(batch.output_file_id)
        raw_response = file_response.text
        reader = jsonlines.Reader(io.StringIO(raw_response))
        responses = []
        for response in reader:
            custom_id = response["custom_id"]
            content = response["response"]["body"]["choices"][0]["message"]["content"]
            keywords_match = re.search("keywords: (.*)", content)
            if keywords_match:
                keywords = [keyword.strip() for keyword in keywords_match.group(1).split(',')]
            else:
                keywords = None
            
            summary_match = re.search("summary: (.*)", content)
            if summary_match:
                summary = summary_match.group(1)
            else:
                summary = None
            
            personalizations = []
            for i in range(MAX_NUM_USERS):
                relevance_match = re.search(f"relevance to researcher {i+1}: (\d), (.*)", content)
                if relevance_match:
                    relevance = int(relevance_match.group(1))
                    relevance_explanation = relevance_match.group(2)
                    personalizations.append({"relevance": relevance, "relevance_explanation": relevance_explanation})
                else:
                    break
            
            response = {
                "arxiv_id": custom_id,
                "keywords": keywords,
                "summary": summary,
                "personalizations": personalizations,
                "raw": file_response.text, 
            }
            responses.append(response)
        return (batch.status, responses)
    else:
        return (batch.status, None)

def batch_manage_llm_submit(date: str, config_file: str, use_cached_file: bool = True):

    with open(config_file, 'r') as f:
        config = json.load(f)
    api_key = config['api_key']
    users = config['users']

    date = datetime.strptime(date, '%Y-%m-%d')
    arxiv_file = ARXIV_ABSTRACTS_PATH_TEMPLATE.format(date.strftime("%Y-%m-%d"))
    if use_cached_file and os.path.exists(arxiv_file):
        with open(arxiv_file, 'r') as f:
            entries = json.load(f)
        print(f'Using cached file {arxiv_file}, loaded {len(entries)} entries.')
    else:
        entries = fetch_arxiv_abstract(catogory='cs.LG', date=date)
        with open(arxiv_file, 'w') as f:
            json.dump(entries, f)
        print(f'Fetched {len(entries)} entries from arXiv on {date.strftime("%Y-%m-%d")}.')
    
    batch, entry_ids = llm_summary_submit(entries, api_key, users)
    
    BATCH_INFO_MANAGE_ONGOING_FILE = 'batch_query_manage/batch_query_manage_ongoing.json'

    with open(BATCH_INFO_MANAGE_ONGOING_FILE, 'r') as f:
        batch_info = json.load(f)

    batch_info[batch.id] = {"input_file_id": batch.input_file_id, "arxiv_list_date": date.strftime("%Y-%m-%d"), "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "entry_ids": entry_ids}

    with open(BATCH_INFO_MANAGE_ONGOING_FILE, 'w') as f:
        json.dump(batch_info, f)
    

def batch_manage_llm_fetch(config_file: str):
    with open(config_file, 'r') as f:
        config = json.load(f)
    api_key = config['api_key']

    BATCH_INFO_MANAGE_ONGOING_FILE = 'batch_query_manage/batch_query_manage_ongoing.json'
    BATCH_INFO_MANAGE_COMPLETED_FILE = 'batch_query_manage/batch_query_manage_completed.json'

    with open(BATCH_INFO_MANAGE_ONGOING_FILE, 'r') as f:
        batch_info = json.load(f)
    
    with open(BATCH_INFO_MANAGE_COMPLETED_FILE, 'r') as f:
        batch_info_completed = json.load(f)

    removed_batch_ids = []
    for batch_id, batch_data in batch_info.items():
        status, response = llm_summary_fetch(api_key=api_key, batch_id=batch_id)
        if status == 'completed':
            batch_info_completed.insert(
                0,
                {
                    "batch_id": batch_id,
                    "arxiv_list_date": batch_data["arxiv_list_date"],
                    "submit_info": batch_data,
                    "response": response,
                    "retrieved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            )
            removed_batch_ids.append(batch_id)
        elif status in ['failed', 'expired', 'cancelling', 'cancelled']:
            with open('batch_query_manage/batch_query_manage_failed.json', 'r') as f:
                batch_info_failed = json.load(f)
            batch_info_failed[batch_id] = {
                    "batch_id": batch_id,
                    "submit_info": batch_data,
                    "status": status,
                    "retrieved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            removed_batch_ids.append(batch_id)
            with open('batch_query_manage/batch_query_manage_failed.json', 'w') as f:
                json.dump(batch_info_failed, f)
        else:
            print(f'Batch {batch_id} is still in progress: {status}')
    
    for batch_id in removed_batch_ids:
        del batch_info[batch_id]
    
    with open(BATCH_INFO_MANAGE_ONGOING_FILE, 'w') as f:
        json.dump(batch_info, f)
    
    with open(BATCH_INFO_MANAGE_COMPLETED_FILE, 'w') as f:
        json.dump(batch_info_completed, f)

