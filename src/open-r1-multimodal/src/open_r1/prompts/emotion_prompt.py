GRPO_PROMPT = '''
### Question
{Question}

### Requirements
First, output the thinking process in <think>...</think> tags. 
    Provide a concise and precise analysis focusing on the critical Action Units (AUs). 
    Mark AU numbers in parentheses, and then explain the connection between the AUs and the emotion. 
    ⚠️Forbidden to use negative or uncertain expressions such as "no", "not", "without" or "maybe".
After the thinking process, output only the final emotion.
    Select one from {Emotions}.
    Use a single word within <answer>...</answer> tags.

### AU Definition
The following are the definitions of Action Units:
    AU1: Inner brow raiser;
    AU2: Outer brow raiser;
    AU4: Brow lowerer;
    AU5: Upper lid raiser;
    AU6: Cheek raiser;
    AU7: Lid tightener;
    AU9: Nose wrinkler;
    AU10: Upper lip raiser;
    AU11: Nasolabial Furrow Deepener;
    AU12: Lip corner puller;
    AU13: Cheek puffer;
    AU14: Dimpler;
    AU15: Lip corner depressor;
    AU16: Lower lip depressor;
    AU17: Chin raiser;
    AU18: Lip pucker;
    AU19: Tongue show;
    AU20: Lip stretcher;
    AU22: Lip funneler;
    AU23: Lip tightener;
    AU24: Lip pressor;
    AU25: Lips parted;
    AU26: Jaw drop;
    AU27: Mouth stretch;
    AU28: Lip suck;
    AU29: Jaw thrust;
    AU30: Jaw sideways;
    AU31: Jaw clencher;
    AU32: Lip bite;
    AU43: Eyes closed.

### Example
Here is some examples of the required analysis style:
<think>The man's eyebrows are pulled together and lowered (AU4), and the mouth corners are downturned (AU15). This combination of brow lowerer (AU4) and lip corner depressor (AU15) is a strong indicator of sadness.</think>
<answer>sadness</answer>

<think>The child's eyebrows are drawn together (AU4), cheeks are slightly raised (AU6), eyelids are tightened (AU7), and there is some nose wrinkling (AU9). The mouth is closed and lacks upward movement indicative of happiness. These action units, particularly AU4, AU6, and AU7, are strong indicators of sadness or distress, as they often appear when a person is feeling unhappy or on the verge of crying.</think>
<answer>sadness</answer>

<think>The woman's cheeks are raised (AU6), the corners of her mouth are pulled upwards (AU12), and her lips are parted (AU25). This classic combination of AU6, AU12, and AU25 is a reliable indicator of genuine happiness or joy, as seen in a Duchenne smile.</think>
<answer>happiness</answer>
'''.strip()

# GRPO_PROMPT = (  
#     "{Question} "  
#     "First output the thinking process in <think>...</think> tags, "  
#     "then output the final emotion—choose only from [anger, happiness, sadness, neutral, disgust, surprise, fear]—using one word in <answer>...</answer> tags."  
# )

SFT_PROMPT = (  
    "{Question} "  
    "Output the final emotion—choose only from [anger, happiness, sadness, neutral, disgust, surprise, fear]."  
)