# Transcript Classification with AssemblyAI's LeMUR API

## Overview
This repo outlines two approaches to zero-shot transcript classification using AssemblyAI's LeMUR API and a given list of topics. The two approaches are:

1. **Chunked Approach**: Process the transcript in smaller excerpts and classify each excerpt separately.
2. **Full Transcript Approach**: Process the entire transcript in one go using the VTT captions file.

## Chunked Approach

#### Advantages:
1. Manages token limits for long transcripts.
2. Allows precise tagging of smaller sections.
3. Enables parallel processing to save time.
4. Easier to debug and refine predictions.
5. Captures dynamic topic shifts accurately.
6. Automatically handles timestamp matching for each excerpt.

#### Disadvantages:
1. Can be more expensive due to multiple API calls and higher token usage.
2. May lose broader context, affecting accuracy for context-dependent trackers.

## Full Transcript Approach

#### Advantages:
1. Cost-efficient for shorter or moderately sized transcripts (fewer API calls).
2. Retains holistic context for better accuracy on context-sensitive trackers.
3. Simpler implementation without chunking logic.

#### Disadvantages:
1. Risk of exceeding token limits for long transcripts.
2. Can introduce noise, reducing precision for specific trackers.
3. Relies on LeMUR to correctly extract exact timestamps from the VTT captions file.

## Recommendation:
- Use chunked approach for long transcripts or when precise tagging is critical. Optimize with larger chunks, selective processing, or limited context overlap.
- Use full transcript approach for shorter transcripts to save costs and maintain full context.