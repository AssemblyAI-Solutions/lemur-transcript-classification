import assemblyai as aai
import json
import re
import pprint

aai.settings.api_key = 'YOUR_API_KEY'

config = aai.TranscriptionConfig()
transcriber = aai.Transcriber(config=config)

transcript = transcriber.transcribe('sales_call.mp3')

trackers = [
    'Greeting',
    'Objections',
    'Competitor',
    'Product',
    'Pricing',
]

trackers_str = '\n'.join(trackers)

answer_format = """
    ```json
    {
        "trackers": ["Competitor", "Objections"]
    }
    ```
"""

prompt_text = f"""
    Label the following phone call transcript excerpt with the most relevant topic trackers from the given <topic_trackers> list.
    If there are no relevant trackers in the excerpt, return an empty list.

    <topic_trackers>
    {trackers_str}
    </topic_trackers>

    Return your answer in the following JSON format. Do not include any other text or comments.

    <answer_format>
    {answer_format}
    </answer_format>
"""

total_input_tokens = 0
total_output_tokens = 0

# Initialize a dictionary to store tracker matches
tracker_matches = {tracker: [] for tracker in trackers}

for excerpt in transcript.get_paragraphs():
    predicted_tags = aai.Lemur().task(
        input_text=excerpt.text,
        final_model=aai.LemurModel.claude3_5_sonnet,
        prompt=prompt_text
    )

    predicted_tags_response = predicted_tags.response
    predicted_tags_usage = predicted_tags.usage

    total_input_tokens += predicted_tags_usage.input_tokens
    total_output_tokens += predicted_tags_usage.output_tokens

    print(excerpt.text)

    extracted_json = re.search(r'```json\n(.*)\n```', predicted_tags_response, re.DOTALL)
    if extracted_json:
        json_predicted_tags = json.loads(extracted_json.group(1))
        pprint.pprint(json_predicted_tags)

        # Update tracker matches with the current excerpt
        for tracker in json_predicted_tags.get("trackers", []):
            tracker_matches[tracker].append({
                "quote": excerpt.text,
                "start_time": excerpt.start,
                "end_time": excerpt.end
            })
    else:
        print("No JSON found in the response.")

# Prepare the final output format
final_output = {
    "trackers": [
        {
            "name": tracker,
            "matches": matches
        } for tracker, matches in tracker_matches.items() if matches
    ]
}

# Print the final output
pprint.pprint(final_output)

print('Total input tokens:', total_input_tokens)
print('Total output tokens:', total_output_tokens)

# Pricing for Claude 3.5 Sonnet
print('Total input cost:', (total_input_tokens/1000)*0.003)
print('Total output cost:', (total_output_tokens/1000)*0.015)