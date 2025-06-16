from abc import ABC, abstractmethod
from collections import defaultdict
import glob
import json
from pathlib import Path
import pathlib
from time import sleep
from typing import List, Optional
import pandas as pd

from pydantic import BaseModel, Field, HttpUrl
from tqdm import tqdm
import os
import requests
from enum import Enum
from ddi_fw.utils import create_folder_if_not_exists


# data = '''
# Lepirudin is a recombinant hirudin formed by 65 amino acids that acts as a highly specific and direct thrombin inhibitor.
# [L41539,L41569] Natural hirudin is an endogenous anticoagulant found in _Hirudo medicinalis_ leeches.
# [L41539] Lepirudin is produced in yeast cells and is identical to natural hirudin except for the absence of sulfate
# on the tyrosine residue at position 63 and the substitution of leucine for isoleucine at position 1 (N-terminal end).[A246609]

# Lepirudin is used as an anticoagulant in patients with heparin-induced thrombocytopenia (HIT),
# an immune reaction associated with a high risk of thromboembolic complications.[A3, L41539]
# HIT is caused by the expression of immunoglobulin G (IgG) antibodies that bind to the complex formed by heparin and platelet factor 4.
# This activates endothelial cells and platelets and enhances the formation of thrombi.
# [A246609] Bayer ceased the production of lepirudin (Refludan) effective May 31, 2012.[L41574]'''

# response = requests.post(url, data=data)

# print(response.content)

HERE = pathlib.Path(__file__).resolve().parent


class NERInterface(ABC):
    """
    An abstract base class to define the interface for Named Entity Recognition (NER).
    """

    @abstractmethod
    def run(self, run_for=[]):
        """
        Run the NER process.
        :param run_for: A list of columns to process.
        """
        pass

class CTakesNER(BaseModel,NERInterface):
    """
    A class to perform Named Entity Recognition (NER) using the cTAKES API.
    Attributes:
        df (pd.DataFrame): The input dataframe containing data to be processed.
        key (str): The key column in the dataframe, default is 'drugbank_id'.
        api_url (str): The URL of the cTAKES API, default is 'http://localhost:8080/ctakes-web-rest/service/analyze?pipeline=Default'.
        output_path (str): The path to save the NER output, default is 'ner-output/ctakes'.
        ids (list): A list of IDs to exclude from processing, default is an empty list.
        columns (list): A list of columns in the dataframe to process, default is an empty list.
    Methods:
        run(run_for=[]):
            Runs the NER process for the specified columns.
        load(filename=None, group=True):
            Loads the NER results from a pickle file.
        create_dataframe(override=False):
            Creates a dataframe from the NER results and saves it as a pickle file.
    """
    
    df: Optional[pd.DataFrame]
    key: str = 'drugbank_id'
    api_url: str = 'http://localhost:8080/ctakes-web-rest/service/analyze?pipeline=Default'
    output_path: str = 'ner-output/ctakes'
    ids: List[str] = Field(default_factory=list)
    columns: List[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True

    def run(self,
            run_for=[]):
        """
        Run the NER process.
        :param run_for: A list of columns to process.
        """
        if self.df is None:
            raise ValueError('Dataframe is not provided')
        for column in self.columns:
            if not os.path.exists(self.output_path+"/"+column):
                os.makedirs(self.output_path+"/"+column)
        for column in self.columns:
            column_output_path = self.output_path+'/'+column
            if not column in run_for:
                continue
            # not include
            if self.ids:
                self.df = self.df[~self.df[self.key].isin(
                    self.ids)]
            for index, row in self.df.iterrows():
                drugbank_id = row[self.key]
                data = row[column]
                # or len(data) == 0:
                if data is None or (isinstance(data, pd.Series) and data.isna().any()) or (isinstance(data, str) and len(data.strip()) == 0):
                # if data is None or pd.isna(data) or (type(data) == str and len(data.strip()) == 0):
                    with open(f'{column_output_path}/{drugbank_id}.json', 'w', encoding='utf-8') as f:
                        json.dump([], f, ensure_ascii=False, indent=4)
                    continue
                data = data.encode()
                response = requests.post(self.api_url, data=data)

                with open(f'{column_output_path}/{drugbank_id}.json', 'w', encoding='utf-8') as f:
                    try:
                        obj = json.loads(response.text)
                        json.dump(obj, f, ensure_ascii=False, indent=4)
                    except:
                        # print(f'{drugbank_id} is not parsable')
                        json.dump([], f, ensure_ascii=False, indent=4)
                        continue

                # if index % 10 == 0:
                #     sleep(10)

    def load(self, filename=None, group=True):
        file_path = filename if filename else HERE.joinpath(
            'output/ctakes/ctakes_ner.pkl')
        df = pd.read_pickle(file_path)

        if group:
            keys = list(df.columns.values)

            df['tui'] = [[]] * df.shape[0]
            df['cui'] = [[]] * df.shape[0]
            df['entities'] = [[]] * df.shape[0]

            tui_columns = [key for key in keys if key.startswith('tui')]
            cui_columns = [key for key in keys if key.startswith('cui')]
            entities_columns = [
                key for key in keys if key.startswith('entities')]
            # bunu tek bir eşitlikle çöz
            df['tui'] = df[tui_columns].values.tolist()
            df['tui'] = df['tui'].apply(
                lambda items: {i for item in items for i in item})

            df['cui'] = df[cui_columns].values.tolist()
            df['cui'] = df['cui'].apply(
                lambda items: {i for item in items for i in item})

            df['entities'] = df[entities_columns].values.tolist()
            df['entities'] = df['entities'].apply(
                lambda items: {i for item in items for i in item})

        return df

    def create_dataframe(self, override=False):  # dataframe_columns=[]
        filename = 'ctakes_ner.pkl'
        if not override and os.path.exists(self.output_path+"/" + filename):
            return self.load(self.output_path+"/" + filename)

        create_folder_if_not_exists(self.output_path+"/" + filename)
        dict_of_dict = defaultdict(dict)
        for column in self.columns:
            all_json_files = f'{self.output_path}/{column}/'+'*.json*'
            for filepath in tqdm(glob.glob(all_json_files)):
                with open(filepath, 'r', encoding="utf8") as f:
                    file_name = Path(f.name).stem
                    t = dict_of_dict[file_name]
                    data = json.load(f)
                    entities = []
                    cuis = []
                    tuis = []
                    if data is None or len(data) == 0:
                        t[self.key] = file_name
                        t[f'cui_{column}'] = []
                        t[f'tui_{column}'] = []
                        t[f'entities_{column}'] = []
                        dict_of_dict[file_name] = t
                        continue
                    for key, value in data.items():
                        entities = [v['text'] for v in value]
                        cuis = [attr['cui']
                                for v in value for attr in v['conceptAttributes']]
                        tuis = [attr['tui']
                                for v in value for attr in v['conceptAttributes']]
                        # codingScheme

                    if self.key not in t:
                        t[self.key] = file_name
                    t[f'cui_{column}'] = cuis
                    t[f'tui_{column}'] = tuis
                    t[f'entities_{column}'] = entities
                dict_of_dict[file_name] = t

        df = pd.DataFrame(dict_of_dict.values(),
                          #   orient='index',
                          #   columns=columns
                          )
        df.to_pickle(self.output_path+"/" + filename)
 
        return df

