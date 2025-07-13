# GRPO_PROMPT = '''
# ### Question
# {Question}

# ### Requirements
# First, output the thinking process in <think>...</think> tags. 
#     - Provide a concise and precise analysis as a continuous paragraph focusing on the critical Action Units (AUs).
#     - Describe the observable facial features with AU numbers in parentheses (e.g., "eyebrows pulled together (AU4)").
#     - After analyzing the possible AUs present in the face, explain how these AUs collectively indicate a specific emotion.
#     - ⚠️Forbidden to use negative or uncertain expressions such as "no", "not", "without" or "maybe".
# After the thinking process, output only the final emotion.
#     - Select one from {Emotions}.
#     - Use a single word within <answer>...</answer> tags.

# ### AU Definition
# The following are the definitions of Action Units:
#     - AU1: Inner brow raiser;
#     - AU2: Outer brow raiser;
#     - AU4: Brow lowerer;
#     - AU5: Upper lid raiser;
#     - AU6: Cheek raiser;
#     - AU7: Lid tightener;
#     - AU9: Nose wrinkler;
#     - AU10: Upper lip raiser;
#     - AU11: Nasolabial Furrow Deepener;
#     - AU12: Lip corner puller;
#     - AU13: Cheek puffer;
#     - AU14: Dimpler;
#     - AU15: Lip corner depressor;
#     - AU16: Lower lip depressor;
#     - AU17: Chin raiser;
#     - AU18: Lip pucker;
#     - AU19: Tongue show;
#     - AU20: Lip stretcher;
#     - AU22: Lip funneler;
#     - AU23: Lip tightener;
#     - AU24: Lip pressor;
#     - AU25: Lips parted;
#     - AU26: Jaw drop;
#     - AU27: Mouth stretch;
#     - AU28: Lip suck;
#     - AU29: Jaw thrust;
#     - AU30: Jaw sideways;
#     - AU31: Jaw clencher;
#     - AU32: Lip bite;
#     - AU43: Eyes closed.

# ### Output Format
# <think>
# [Description of observable facial features with AU numbers in parentheses]
# [Explanation of how these AUs relate to a specific emotion]
# </think>
# <answer>[single emotion word]</answer>

# ### Example
# Here are some examples that meet the requirements, just refer to the reasoning format of "AU first, then emotion" rather than the content:
# <think>The man's eyebrows are pulled together and lowered (AU4), and the mouth corners are downturned (AU15). This combination of brow lowerer (AU4) and lip corner depressor (AU15) is a strong indicator of sadness.</think>
# <answer>sadness</answer>

# <think>The child's eyebrows are drawn together (AU4), cheeks are slightly raised (AU6), eyelids are tightened (AU7), and there is some nose wrinkling (AU9). The mouth is closed and lacks upward movement indicative of happiness. These action units, particularly AU4, AU6, and AU7, are strong indicators of sadness or distress, as they often appear when a person is feeling unhappy or on the verge of crying.</think>
# <answer>sadness</answer>

# <think>The woman's cheeks are raised (AU6), the corners of her mouth are pulled upwards (AU12), and her lips are parted (AU25). This classic combination of AU6, AU12, and AU25 is a reliable indicator of genuine happiness or joy, as seen in a Duchenne smile.</think>
# <answer>happiness</answer>

# ### Requirements
# First, output the thinking process in <think>...</think> tags. 
#     - Provide a concise and precise analysis as a continuous paragraph focusing on the critical Action Units (AUs).
#     - Describe the observable facial features with AU numbers in parentheses (e.g., "eyebrows pulled together (AU4)").
#     - After analyzing the possible AUs present in the face, explain how these AUs collectively indicate a specific emotion.
#     - ⚠️Forbidden to use negative or uncertain expressions such as "no", "not", "without" or "maybe".
# After the thinking process, output only the final emotion.
#     - Select one from {Emotions}.
#     - Use a single word within <answer>...</answer> tags.
# '''.strip()

EXAMPLES = '''
### Example
Here are some examples that meet the requirements, just refer to the reasoning format of "AU first, then emotion" rather than the content:
<think>The man's eyebrows are pulled together and lowered (AU4), and the mouth corners are downturned (AU15). This combination of brow lowerer (AU4) and lip corner depressor (AU15) is a strong indicator of sadness.</think>
<answer>sadness</answer>

<think>The child's eyebrows are drawn together (AU4), cheeks are slightly raised (AU6), eyelids are tightened (AU7), and there is some nose wrinkling (AU9). The mouth is closed and lacks upward movement indicative of happiness. These action units, particularly AU4, AU6, and AU7, are strong indicators of sadness or distress, as they often appear when a person is feeling unhappy or on the verge of crying.</think>
<answer>sadness</answer>

<think>The woman's cheeks are raised (AU6), the corners of her mouth are pulled upwards (AU12), and her lips are parted (AU25). This classic combination of AU6, AU12, and AU25 is a reliable indicator of genuine happiness or joy, as seen in a Duchenne smile.</think>
<answer>happiness</answer>
'''.strip()


GRPO_PROMPT = (
    "Question: {Question}\n"
    "First, output the thinking process in <think>...</think> tags, "
    "producing a concise and precise analysis by only describing the most decisive AUs and how these lead to your emotion inference.\n"
    "Seamlessly integrate only truly relevant AU numbers (in parentheses), directly tying each to your emotional reasoning—avoid uncertainty, negations, or mentioning unlikely emotions.\n"
    "After the thinking process, output only the final emotion—choose one from {Emotions}—as a single word in <answer>...</answer> tags.\n\n"
    "Use these AU definitions and their possible emotional associations:\n"
    "AU1: Inner brow raiser; "
    "AU2: Outer brow raiser; "
    "AU4: Brow lowerer; "
    "AU5: Upper lid raiser; "
    "AU6: Cheek raiser; "
    "AU7: Lid tightener; "
    "AU9: Nose wrinkler; "
    "AU10: Upper lip raiser; "
    "AU11: Nasolabial Furrow Deepener; "
    "AU12: Lip corner puller; "
    "AU13: Cheek puffer; "
    "AU14: Dimpler; "
    "AU15: Lip corner depressor; "
    "AU16: Lower lip depressor; "
    "AU17: Chin raiser; "
    "AU18: Lip pucker; "
    "AU19: Tongue show; "
    "AU20: Lip stretcher; "
    "AU22: Lip funneler; "
    "AU23: Lip tightener; "
    "AU24: Lip pressor; "
    "AU25: Lips parted; "
    "AU26: Jaw drop; "
    "AU27: Mouth stretch; "
    "AU28: Lip suck; "
    "AU29: Jaw thrust; "
    "AU30: Jaw sideways; "
    "AU31: Jaw clencher; "
    "AU32: Lip bite; "
    "AU43: Eyes closed.\n\n"
    "Here is a sample of the required analysis style:\n"
    "<think>In analyzing the facial expression, key Action Units include upper lip raising (AU10), chin raising (AU17), and lip funneling (AU22). The upper lip raised indicates a display of disdain or hostility, while the chin raised suggests a feeling of defiance or assertiveness. The lip funneling can indicate tension or frustration in the expression. Together, these AUs coalesce to portray a strong emotional state where the individual might feel anger or strong irritation. The combination of disdain, defiance, and tension clearly signifies an intense emotional reaction, thus pointing towards anger as the primary emotion.</think>\n"
    "<answer>anger</answer>"
)

GT = '''
### Ground Truth
Your analysis MUST identify these specific Action Units: {true_aus}
And your final answer MUST be this exact emotion: {true_emotion}
'''.strip()
INFER_PROMPT = GRPO_PROMPT + GT

# GRPO_PROMPT = (  
#     "{Question} "  
#     "First output the thinking process in <think>...</think> tags, "  
#     "then output the final emotion—choose only from [anger, happiness, sadness, neutral, disgust, surprise, fear]—using one word in <answer>...</answer> tags."  
# )

SFT_PROMPT = (  
    "{Question} "  
    "Output the final emotion—choose only from [anger, happiness, sadness, neutral, disgust, surprise, fear]."  
)