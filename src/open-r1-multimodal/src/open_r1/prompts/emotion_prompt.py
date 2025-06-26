GRPO_PROMPT = (  
    "{Question} "  
    "First output the thinking process in <think>...</think> tags, "  
    "then output the final emotion—choose only from [anger, happiness, sadness, neutral, disgust, surprise, fear]—using one word in <answer>...</answer> tags."  
)

SFT_PROMPT = (  
    "{Question} "  
    "Output the final emotion—choose only from [anger, happiness, sadness, neutral, disgust, surprise, fear]."  
)