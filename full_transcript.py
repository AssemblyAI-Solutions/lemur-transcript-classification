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
    'Pricing'
]

trackers_str = '\n'.join(trackers)

answer_format = """
    ```json
    {
        "trackers": 
        [{
            "name": "Greeting",
            "matches": [
                {
                    "quote": "Thank you for calling.",
                    "start_time": 100,
                    "end_time": 200
                }
            ]
        }]
    }
    ```
"""

prompt_text = f"""
    Classify the following phone call transcript into the most relevant topic trackers from the given <topic_trackers> list.
    If there are no relevant trackers in the transcript, return an empty list.

    <topic_trackers>
    {trackers_str}
    </topic_trackers>

    For each topic tracker, identify the most relevant excerpts from the transcript that align with the topic tracker's intent. Provide the exact quote and timestamps for each match.
    If there are no matches for a tracker, do not include it in the response.

    Return your answer in valid JSON in the given format.
    Do not include any other text or comments.
    start_time and end_time need to be in milliseconds.
    
    <answer_format>
    {answer_format}
    </answer_format>
"""

predicted_tags = aai.Lemur().task(
    input_text=transcript.export_subtitles_vtt(),
    final_model=aai.LemurModel.claude3_5_sonnet,
    prompt=prompt_text
)

predicted_tags_response = predicted_tags.response
predicted_tags_usage = predicted_tags.usage

extracted_json = re.search(r'```json\n(.*)\n```', predicted_tags_response, re.DOTALL)
if extracted_json:
    json_predicted_tags = json.loads(extracted_json.group(1))
    pprint.pprint(json_predicted_tags)
else:
    print("No JSON found in the response.")

print('Total input tokens:', predicted_tags_usage.input_tokens)
print('Total output tokens:', predicted_tags_usage.output_tokens)

# Pricing for Claude 3.5 Sonnet
print('Total input cost:', (predicted_tags_usage.input_tokens/1000)*0.003)
print('Total output cost:', (predicted_tags_usage.output_tokens/1000)*0.015)