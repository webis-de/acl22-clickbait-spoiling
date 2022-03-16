import argparse
import sys
import json
import csv
import string
import unicodedata

from Dataloader import loadDataSplit, load_json_asdict


def rearrange(data_df, data_json_path):
    # Load json of data
    data_json = load_json_asdict(data_json_path, 'uuid')

    sq_data_trans = {
        "data": [],
        "version": "1.1"
    }

    for index in range(0, data_df.shape[0]):
        # Read the documents
        data = data_df.iloc[index]
        uuid = str(data['uuid'])

        context = ''

        sq_data = {
            "paragraphs": [],
            "title": data_json[uuid]['targetTitle']
        }

        context += data_json[uuid]['targetTitle'] + ' -'

        for i, paragraph in enumerate(data_json[uuid]['targetParagraphs']):
            paratext = unicodedata.normalize("NFKD", paragraph)

            context += ' '+paratext

        sq_para = {
            "context": context,
            "qas": [{
                "answers": [{
                    "answer_start": context.find(data_json[uuid]["spoiler"][0]),
                    "text": data_json[uuid]["spoiler"][0]
                }],
                "id": data_json[uuid]["uuid"],
                "question": data_json[uuid]["postText"][0]
            }]
        }
        sq_data["paragraphs"].append(sq_para)

        sq_data_trans["data"].append(sq_data)

    return sq_data_trans

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outfile", help="file to write transformed data to")
    parser.add_argument("--infile", help="CBS20 file directory")
    parser.add_argument("--tag", help="tag of class to transform")
    args = parser.parse_args()

    data = loadDataSplit(args.infile)
    if args.tag:
        run_data = data[data['label'] == args.tag]
    else:
        print('No --tag <value> given --> Transforming passage and phrase data')
        run_data = data[data['label'] != 'multi']

    print(run_data.shape)

    arranged_data = rearrange(run_data, args.infile)

    with open(args.outfile, 'w', encoding='utf8') as ofile:
        json.dump(arranged_data, ofile, ensure_ascii=False, indent=1)
        ofile.write('\n')


if __name__ == "__main__":
    main()