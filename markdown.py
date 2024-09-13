import json
import os

date = '2024-09-12'

CONFIG_FILE = 'config.json'
with open(CONFIG_FILE, 'r') as f:
    config = json.load(f)

arxiv_file = f'arxiv_abstracts/{date}.json'
with open(arxiv_file, 'r') as f:
    arxiv_list = json.load(f)

arxiv_list_dict = {}
for entry in arxiv_list:
    arxiv_list_dict[entry['id'].removeprefix('http://arxiv.org/abs/')] = {
        "title": entry['title'].replace('\n', ' ').replace('\r', ''),
        "authors": [author['name'] for author in entry['authors']],
        "summary": entry['summary'],
        "link": entry['link'],
        "tags": entry['tags']
    }

BATCH_INFO_MANAGE_COMPLETED_FILE = 'batch_query_manage/batch_query_manage_completed.json'

with open(BATCH_INFO_MANAGE_COMPLETED_FILE, 'r') as f:
    batch_info_completed = json.load(f)

cur_day_entries = None
for batch_data in batch_info_completed:
    if batch_data['arxiv_list_date'] == date:
        cur_day_entries = batch_data['response']
        break

if cur_day_entries is None:
    print(f'No entries found for {date}')
    exit(1)

user_ids = [user["id"] for user in config['users']]
for i, id in enumerate(user_ids):
    markdown_dir = f'markdown/{id}'
    os.makedirs(markdown_dir, exist_ok=True)
    markdown_file = f'markdown/{id}/{date}.md'
    cur_day_entries = sorted(cur_day_entries, key=lambda x: x['personalizations'][i]['relevance'], reverse=True)
    with open(markdown_file, 'w', encoding='utf-8') as f:
        for entry in cur_day_entries:
            arxiv_info = arxiv_list_dict[entry['arxiv_id']]
            f.write(f'# [{arxiv_info["title"]}]({arxiv_info["link"]})\n')
            f.write(f"- Authors: {', '.join(arxiv_info['authors'])}\n")
            f.write(f"- Keywords: {', '.join(entry['keywords'])}\n")
            f.write(f"- Relevance: {entry['personalizations'][i]['relevance']}\n\n")
            f.write(f"  {entry['personalizations'][i]['relevance_explanation']}\n")
            f.write(f'- Summary\n\n')
            f.write(f'  {entry["summary"]}\n')
