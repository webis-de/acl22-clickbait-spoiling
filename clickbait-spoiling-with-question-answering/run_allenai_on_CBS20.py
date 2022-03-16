import argparse
from os.path import isfile

import re
import numpy as np
import tensorflow as tf
import json

from docqa.data_processing.document_splitter import MergeParagraphs, TopTfIdf, ShallowOpenWebRanker, PreserveParagraphs
from docqa.data_processing.qa_training_data import ParagraphAndQuestion, ParagraphAndQuestionSpec
from docqa.data_processing.text_utils import NltkAndPunctTokenizer, NltkPlusStopWords
from docqa.doc_qa_models import ParagraphQuestionModel
from docqa.model_dir import ModelDir
from docqa.utils import flatten_iterable

from Dataloader import loadDataSplit, load_json_asdict

"""
Script to run a model on user provided question/context document. 
This demonstrates how to use our document-pipeline on new input
"""

def allenaiqa_run(modeldir, data_df, data_json_path):
        print("Preprocessing...")

        # Load json of data
        data_json = load_json_asdict(data_json_path, 'uuid')

        # Load the model
        model_dir = ModelDir(modeldir)
        model = model_dir.get_model()
        if not isinstance(model, ParagraphQuestionModel):
            raise ValueError("This script is built to work for ParagraphQuestionModel models only")

        print('data to spoil:\n', data_df.shape)
        for index in range(0, data_df.shape[0]):
            print('index', index)
            # Read the documents
            data = data_df.iloc[index]
            uuid = str(data['uuid'])

            documents = [data_json[uuid]['targetTitle']] + [data_json[uuid]['targetParagraphs']]

            # Tokenize the input, the models expects data to be tokenized using `NltkAndPunctTokenizer`
            # Note the model expects case-sensitive input
            tokenizer = NltkAndPunctTokenizer()
            question = tokenizer.tokenize_paragraph_flat(data['clickbait'])  # List of words
            # Now list of document->paragraph->sentence->word
            documents = [[tokenizer.tokenize_paragraph(p) for p in doc] for doc in documents]

            # Now group the document into paragraphs, this returns `ExtractedParagraph` objects
            # that additionally remember the start/end token of the paragraph within the source document
            splitter = MergeParagraphs(400)
            # splitter = PreserveParagraphs() # Uncomment to use the natural paragraph grouping
            documents = [splitter.split(doc) for doc in documents]

            # Now select the top paragraphs using a `ParagraphFilter`
            if len(documents) == 1:
                # Use TF-IDF to select top paragraphs from the document
                selector = TopTfIdf(NltkPlusStopWords(True), n_to_select=5)
                context = selector.prune(question, documents[0])
            else:
                # Use a linear classifier to select top paragraphs among all the documents
                selector = ShallowOpenWebRanker(n_to_select=10)
                context = selector.prune(question, flatten_iterable(documents))

            print("Select %d paragraph" % len(context))

            if model.preprocessor is not None:
                # Models are allowed to define an additional pre-processing step
                # This will turn the `ExtractedParagraph` objects back into simple lists of tokens
                context = [model.preprocessor.encode_text(question, x) for x in context]
            else:
                # Otherwise just use flattened text
                context = [flatten_iterable(x.text) for x in context]

            print("Setting up model")
            # Tell the model the batch size (can be None) and vocab to expect, This will load the
            # needed word vectors and fix the batch size to use when building the graph / encoding the input
            voc = set(question)
            for txt in context:
                voc.update(txt)
            model.set_input_spec(ParagraphAndQuestionSpec(batch_size=len(context)), voc)

            # Now we build the actual tensorflow graph, `best_span` and `conf` are
            # tensors holding the predicted span (inclusive) and confidence scores for each
            # element in the input batch, confidence scores being the pre-softmax logit for the span
            print("Build tf graph")
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            # We need to use sess.as_default when working with the cuNND stuff, since we need an active
            # session to figure out the # of parameters needed for each layer. The cpu-compatible models don't need this.
            with sess.as_default():
                # 8 means to limit the span to size 8 or less
                best_spans, conf = model.get_prediction().get_best_span(8)

            # Loads the saved weights
            model_dir.restore_checkpoint(sess)

            # Now the model is ready to run
            # The model takes input in the form of `ContextAndQuestion` objects, for example:
            data = [ParagraphAndQuestion(x, question, None, "user-question%d" % i)
                    for i, x in enumerate(context)]

            print("Starting run" + " " + str(index+1))

            # The model is run in two steps, first it "encodes" a batch of paragraph/context pairs
            # into numpy arrays, then we use `sess` to run the actual model get the predictions
            encoded = model.encode(data, is_train=False)  # batch of `ContextAndQuestion` -> feed_dict
            best_spans, conf = sess.run([best_spans, conf], feed_dict=encoded)  # feed_dict -> predictions

            best_para = np.argmax(conf)  # We get output for each paragraph, select the most-confident one to print

            output = 'CBS20_allenai_answers.jsonl'

            with open(output, 'a') as o:
                res = {
                    'uuid': uuid,
                    'allenai_answer': " ".join(
                        context[best_para][best_spans[best_para][0]:best_spans[best_para][1] + 1]),
                    'spoiler': data_json[uuid]['spoiler'],
                    'allenai_confidence': str(conf[best_para]),
                    'allenai_best_paragraph': str(best_para),
                    'allenai_best_span': str(best_spans[best_para])
                }
                json.dump(res, o)
                o.write('\n')

            tf.get_variable_scope().reuse_variables()

def main():
    parser = argparse.ArgumentParser(description="Run an ELMo model on CBS20")
    parser.add_argument("--model", help="Model directory")
    parser.add_argument("--infile", help="CBS20 file directory")
    parser.add_argument('--tag', choices=['phrase', 'passage'],
                        help="class of clickbaits to spoil")
    args = parser.parse_args()

    data = loadDataSplit(args.infile)
    run_data = data[data['label'] == args.tag]

    print(run_data.shape)
    print(run_data.head())

    allenaiqa_run(args.model, run_data, args.infile)

if __name__ == "__main__":
    main()