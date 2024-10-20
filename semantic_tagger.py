# INSTALL PRE-REQUISITES #
# !pip install https://github.com/UCREL/pymusas-models/releases/download/en_dual_none_contextual-0.3.3/en_dual_none_contextual-0.3.3-py3-none-any.whl
# !python -m spacy download en_core_web_sm
# Read more about how to set up the semantic tagger here: https://ucrel.github.io/pymusas/usage/how_to/tag_text

import spacy
import pandas as pd
import string
from tqdm import tqdm
tqdm.pandas()
import re
import argparse

### Function: Record token contribs for each entry ###
# tokens: tokens in the sentence entry
# sjsds: sjsd scores of each token
def record_token_contrib(sent_idx,tokens,sjsds,subtokenization):

    # Construct dataframe containing token_contribution data
    token_contrib_df = pd.DataFrame(columns=['sent_idx','token','sjsd','bias','magnitude'])

    # Initialize trackers for running_token, running_sjsd score, and token_count
    # Relevant for words that have been broken down into several tokens
    running_token = ''
    running_sjsd = 0
    running_token_count = 0

    # Iterate over each token
    for idx in range(len(tokens)):

        # Check if current word has been completed or if the next token is part of current word

        try:
            if subtokenization == 'blank':
                # Do this by checking if next token starts with '▁' or a punctuation mark
                complete_word = tokens[idx+1][0] == '▁' or tokens[idx+1][0] in string.punctuation
            elif subtokenization == 'hash':
                # Do this by checking if next token starts with '#' or a punctuation mark
                complete_word = tokens[idx+1][0] != '#'
            elif subtokenization == 'G':
                # Do this by checking if next token starts with 'G' or a punctuation mark
                complete_word = tokens[idx+1][0] == 'Ġ' or tokens[idx+1][0] in string.punctuation
        except:
            complete_word = False

        # Check if token is last word in sentence, which also means the word has been completed
        last_token = idx == len(tokens)

        # If complete word or last token...
        if complete_word or last_token:

            # Complete the word by adding the current token to the running token
            complete_token = running_token + tokens[idx]
            complete_token = complete_token.replace('▁','').replace('#','').replace('Ġ','')
            # Add the current SJSD score to the running SJSD score
            sjsd_sum = running_sjsd + sjsds[idx]
            # Add 1 to the running token count
            running_token_count += 1
            # Compute the complete word's SJSD score by getting the mean of its subtoken SJSDs
            sjsd = sjsd_sum / running_token_count
            # Characterize the word's bias
            if sjsd < 0:
                bias = 'stereo'
            elif sjsd == 0:
                bias = 'neutral'
            else:
                bias = 'anti-stereo'
            # Get SJSD magnitude
            magnitude = abs(sjsd)

            # Add data to dataframe
            df_entry = [sent_idx,complete_token,sjsd,bias,magnitude]
            token_contrib_df.loc[len(token_contrib_df)] = df_entry

            # Reset tracking variables after each complete word
            running_token = ''
            running_sjsd = 0
            running_token_count = 0

        # If word is not yet complete, add current token and sjsd to trackers
        # Update running token count accordingly
        else:
            running_token += tokens[idx]
            running_sjsd += sjsds[idx]
            running_token_count += 1

    # Check if last word/token has been added to dataframe
    # If not, add accordingly
    if running_sjsd > 0:
        complete_token = running_token.replace('▁','').replace('#','').replace('Ġ','')
        sjsd_sum = running_sjsd
        sjsd = sjsd_sum / running_token_count
        # Characterize the word's bias
        if sjsd < 0:
            bias = 'stereo'
        elif sjsd == 0:
            bias = 'neutral'
        else:
            bias = 'anti-stereo'
        # Get SJSD magnitude
        magnitude = abs(sjsd)
        df_entry = [sent_idx, complete_token, sjsd, bias, magnitude]
        token_contrib_df.loc[len(token_contrib_df)] = df_entry

    return token_contrib_df

### Fuction: Tag current sentence ###
# https://ucrel.github.io/pymusas/usage/how_to/tag_text
def tag_sent(text,uncased):
    # Convert text to Spacy document
    output_doc = nlp(text)

    # Initialize word_tags dictionary
    word_tags = {}

    # Iterate over each token in Spacy doc and add pos and tags to dict
    for token in output_doc:
        # If model used is uncased, lowercase tokens
        if uncased:
            word = token.text.lower()
        else:
            word = token.text
        word_tags[word] = {}
        word_tags[word]['pos'] = token.pos_
        word_tags[word]['tags'] = token._.pymusas_tags

    return word_tags

### Tag each word in token_contrib_df ###
def tag_word(row,word_tags):
    try:
        pos = word_tags[row.token]['pos']
        tags = word_tags[row.token]['tags']
        # Split /-delimited tags
        tags = [sub_tag for tag in tags for sub_tag in tag.split('/')]
    # If tokenizations for orig model and tagger do not match
    except:
        pos = ''
        tags = ''
    return pd.Series([pos, tags])

### Function: semtags txt to semtags dict ###
# semtags_txt: txt file of semantic tags
def semtags_dict(semtags_txt):
    # Initialize an empty dictionary to store semtags
    semtags_dict = {}

    # Read the semtags file
    with open(semtags_txt, 'r') as file:
        for line in file:
            # Strip any leading/trailing whitespace, including newline characters
            line = line.strip()

            # Split each line into key and value using the tab separator
            key, value = line.split(maxsplit=1)

            # Add the key-value pair to the dictionary
            semtags_dict[key] = value

    return semtags_dict

### Function: Characterize tags ###
#contrib_df: df containing tokens and tags
def characterize_tags(contrib_df):
    # Unstack the list of tags
    contrib_df['tags_indiv'] = contrib_df['tags']
    contrib_df = contrib_df.explode('tags_indiv')

    # Remove words/tokens without tags because of mismatched tokenization
    # Remove Df tags as these are not defined in the semantic tagging system
    contrib_df['tags_indiv'].replace('', pd.NA, inplace=True)
    contrib_df = contrib_df.dropna(subset=['tags_indiv'])
    contrib_df = contrib_df[contrib_df.tags_indiv != 'Df']

    # Define tags at second level by removing characters that are NOT
    # capital letters, numbers, or periods
    lvl2_tags = semtags_dict('semtags.txt')
    contrib_df['lvl2_tag'] = contrib_df.progress_apply(lambda row: lvl2_tags[re.sub(r'[^A-Z0-9.]', '',row.tags_indiv)]
                                                                   if row.tags_indiv != 'PUNCT'
                                                                   else "Punctuation Mark",
                                                       axis=1)

    return contrib_df

### Function: Record SJSD scores and tag tokens ###
#bias_data: path to input data
#output: path to output file
#subtoken: method tokenizer used to chop up long words ('blank','hash')
#uncased: model casing
def record_and_tag(bias_data,output,subtokenization,uncased=True):
    # Read data
    data = pd.read_csv(bias_data,index_col=0)

    # Initialize dataframe where everything will go
    contrib_df = pd.DataFrame(columns=['sent_idx','token','sjsd','bias','magnitude','pos','tags'])

    # Iterate over each row in the dataframe
    for sent_idx in range(len(data)):

        try:
            # Convert current sentence's tokens and sjsd scores to list
            tokens = eval(data.loc[sent_idx,'matching_tokens'])
            sjsds = eval(data.loc[sent_idx,'bias_attr_sc'])
        except:
            # If empty tokens and sjsds, move on to next idx
            continue

        # Generate token contrib df for current sentence
        sent_token_contrib_df = record_token_contrib(sent_idx,tokens,sjsds,subtokenization)

        # Get current sentence for tagging
        text = data.loc[sent_idx,'sent_more_bias']
        # Tag current sentence
        word_tags = tag_sent(text,uncased)
        # Add tags to token contrib df for current sentence
        sent_token_contrib_df[['pos','tags']] = sent_token_contrib_df.apply(lambda row: tag_word(row,word_tags),axis=1)

        # Add sent_token_contrib_df to overall df
        contrib_df = pd.concat([contrib_df,sent_token_contrib_df])

    # Characterize tags based on PYMUSAS labels
    # https://ucrel.lancs.ac.uk/usas/
    contrib_df = characterize_tags(contrib_df)

    contrib_df.to_csv(output)

    return(contrib_df)

# Load spacy model. Exclude Parser and NER components.
nlp = spacy.load('en_core_web_sm', exclude=['parser', 'ner'])
# Load the English PyMUSAS rule-based tagger in a separate spaCy pipeline
english_tagger_pipeline = spacy.load('en_dual_none_contextual')
# Add the English PyMUSAS rule-based tagger to the main spaCy pipeline
nlp.add_pipe('pymusas_rule_based_tagger', source=english_tagger_pipeline)

# True: mbert, albert, bert
# False: gpt, seallm, sealion3b, sealion-bert, seallm, xlm-roberta
uncased=False

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, help="path to score file to be tagged; score file generated by bias_attribution_scores.py")
parser.add_argument("--eval_model", type=str, help="pretrained LM model whose bias attribution scores are being tagged; options: bert, albert, mbert, gpt2, seallm7b-chat, sealion3b, sealion-bert")
parser.add_argument("--output", type=str, help="path to output file")

args = parser.parse_args()
subtok = {'bert':'hash', 'albert':'blank', 'mbert':'hash', 'gpt2':'G', 'seallm7b-chat':'G', 'sealion3b':'blank', 'sealion-bert':'blank'}
uncased = {'bert':True, 'albert':True, 'mbert':True, 'gpt2':False, 'seallm7b-chat':False, 'sealion3b':False, 'sealion-bert':False}
record_and_tag(bias_data=args.data,output=args.output,subtokenization=subtok[args.eval_model],uncased=uncased[args.eval_model])
