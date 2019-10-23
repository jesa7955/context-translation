import glob
import logging
import json
import os

import luigi
import gokart
from sklearn.model_selection import train_test_split

logger = logging.getLogger('luigi-interface')


class MergeJijiFiles(gokart.TaskOnKart):
    task_namespace = "context_nmt"
    source_path = luigi.Parameter()
    dev_proportion = luigi.FloatParameter()
    test_proportion = luigi.FloatParameter()
    score_threhold = luigi.FloatParameter()
    quality_aware = luigi.BoolParameter()
    score_threhold = luigi.FloatParameter()

    def requires(self):
        pass

    def output(self):
        return self.make_target("merged_jiji_splits.pkl")

    def run(self):
        documents = {}
        document = []
        instance = {}
        last_doc_name = None
        for file_path in glob.glob(self.source_path + '/*.txt'):
            with open(file_path) as source:
                lines = source.readlines()
            for line in lines:
                if line[0] == '#':
                    _, doc_name, score = line.split()
                    doc_name = doc_name.split('_')[0]
                    score = float(score.split('=')[1])
                    if doc_name != last_doc_name and document:
                        documents[last_doc_name] = document
                        document.clear()
                    else:
                        if instance and (not self.quality_aware
                                         or score >= self.score_threhold):
                            document.append(instance)
                        instance = {'en': '', 'ja': '', 'score': float(score)}
                    last_doc_name = doc_name
                elif line[:3] == 'ja:':
                    instance['ja'] = line.split('ja:')[1].strip()
                elif line[:3] == 'en:':
                    instance['en'] = line.split('en:')[1].strip()
            documents[last_doc_name] = document
        logger.info(f"There are {len(documents)} documents")
        test_size = self.test_proportion
        dev_size = self.dev_proportion / (1 - test_size)
        train, test = map(
            dict, train_test_split(list(documents.items()),
                                   test_size=test_size))
        train, dev = map(
            dict, train_test_split(list(train.items()), test_size=dev_size))
        self.dump({'train': train, 'dev': dev, 'test': test})


class GenerateJijiDataSplits(gokart.TaskOnKart):
    task_namespace = "context_nmt"
    jiji_source_path = luigi.Parameter()
    target_path = luigi.Parameter()
    dev_proportion = luigi.FloatParameter()
    test_proportion = luigi.FloatParameter()
    quality_aware = luigi.BoolParameter()
    score_threhold = luigi.FloatParameter(default=0.8)

    def requires(self):
        return MergeJijiFiles(source_path=self.jiji_source_path,
                              dev_proportion=self.dev_proportion,
                              test_proportion=self.test_proportion,
                              quality_aware=self.quality_aware,
                              score_threhold=self.score_threhold)

    def output(self):
        return self.input()

    def run(self):
        data = self.load()
        if not os.path.isdir(self.target_path):
            os.mkdir(self.target_path)
        for name, split in data.items():
            with open(self.target_path + f'/{name}.json', 'w') as target:
                json.dump(split, target, ensure_ascii=False)
