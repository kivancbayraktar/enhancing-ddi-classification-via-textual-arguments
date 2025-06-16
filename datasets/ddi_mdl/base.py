import pathlib
from typing import Any, List, Optional, Tuple
from datasets.core import BaseDataset, TextDatasetMixin, generate_sim_matrices_new, generate_vectors
from datasets.db_utils import create_connection
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator, root_validator
from abc import ABC, abstractmethod
from sklearn.preprocessing import LabelBinarizer
import logging

from ner.ner import CTakesNER


try:
    from ddi_fw.vectorization import IDF
except ImportError:
    raise ImportError(
        "Failed to import vectorization module. Ensure that the module exists and is correctly installed. ")

logger = logging.getLogger(__name__)

# Constants for embedding, chemical properties, and NER columns
LIST_OF_EMBEDDING_COLUMNS = [
    'all_text', 'description', 'synthesis_reference', 'indication',
    'pharmacodynamics', 'mechanism_of_action', 'toxicity', 'metabolism',
    'absorption', 'half_life', 'protein_binding', 'route_of_elimination',
    'volume_of_distribution', 'clearance'
]

LIST_OF_CHEMICAL_PROPERTY_COLUMNS = ['enzyme', 'target', 'pathway', 'smile']
LIST_OF_NER_COLUMNS = ['tui', 'cui', 'entities']

HERE = pathlib.Path(__file__).resolve().parent


class DDIMDLDataset(BaseDataset,TextDatasetMixin):
    dataset_name: str = "DDIMDLDataset"
    # index_path: str = Field(default_factory=lambda: str(
    #     pathlib.Path(__file__).resolve().parent.joinpath('indexes')))
    # drugs_df: pd.DataFrame = Field(default_factory=pd.DataFrame)
    # ddis_df: pd.DataFrame = Field(default_factory=pd.DataFrame)
    drugs_df: Optional[pd.DataFrame] = None
    ddis_df: Optional[pd.DataFrame] = None

    chemical_property_columns: list[str] = Field(
        default_factory=lambda: LIST_OF_CHEMICAL_PROPERTY_COLUMNS)
    embedding_columns: list[str] = Field(default_factory=list)
    ner_columns: list[str] = Field(default_factory=list)
    ner_df: pd.DataFrame | None = None
    tui_threshold: float | None = None
    cui_threshold: float | None = None
    entities_threshold: float | None = None
    _ner_threshold: dict[str,Any] |None = None

    # @model_validator

    def validate_columns(self, values):
        if not set(values['chemical_property_columns']).issubset(LIST_OF_CHEMICAL_PROPERTY_COLUMNS):
            raise ValueError("Invalid chemical property columns")
        if not set(values['ner_columns']).issubset(LIST_OF_NER_COLUMNS):
            raise ValueError("Invalid NER columns")
        return values

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.index_path = str(
            pathlib.Path(__file__).resolve().parent.joinpath('indexes'))
        # self.additional_config = kwargs.get('dataset_additional_config', {})
        if self.additional_config:
            ner = self.additional_config.get('ner', {})
            ner_data_file = ner.get('data_file', None)
            self._ner_threshold = ner.get('thresholds', None)
            # if self.ner_threshold:
            #     for k, v in self.ner_threshold.items():
            #         kwargs[k] = v
        
            self.ner_df = CTakesNER(df=None).load(
                filename=ner_data_file) if ner_data_file else None
        
        columns = kwargs['columns']
        if columns:
            chemical_property_columns = []
            embedding_columns = []
            ner_columns = []
            for column in columns:
                if column in LIST_OF_CHEMICAL_PROPERTY_COLUMNS:
                    chemical_property_columns.append(column)
                elif column in LIST_OF_EMBEDDING_COLUMNS:
                    embedding_columns.append(column)
                elif column in LIST_OF_NER_COLUMNS:
                    ner_columns.append(column)
                else:
                    raise Exception(f"{column} is not related this dataset")
        
            self.chemical_property_columns = chemical_property_columns
            self.embedding_columns = embedding_columns 
            self.ner_columns = ner_columns 
            self.columns = [] # these variable is modified in prep method
        
        self.class_column = 'event_category'
        _db_path = HERE.joinpath('data/event.db')

        self.__similarity_related_columns__ = []
        self.__similarity_related_columns__.extend(
            self.chemical_property_columns)
        self.__similarity_related_columns__.extend(self.ner_columns)
        # TODO with resource
        self._conn = create_connection(_db_path.absolute().as_posix())
        # self.load_drugs_and_events()
        logger.info(f'{self.dataset_name} is initialized')

    def load_drugs_and_events(self):
        self.drugs_df = self.__select_all_drugs_as_dataframe()
        self.ddis_df = self.__select_all_events()

    def __select_all_drugs_as_dataframe(self):
        headers = ['index', 'id', 'name',
                   'target', 'enzyme', 'pathway', 'smile']
        if self._conn is None:
            raise Exception("There is no connection")
        cur = self._conn.cursor()
        cur.execute(
            '''SELECT "index", id, name, target, enzyme, pathway, smile FROM drug'''
        )
        rows = cur.fetchall()
        df = pd.DataFrame(columns=headers, data=rows)

        # Convert string fields to lists
        for col in ['enzyme', 'target', 'pathway', 'smile']:
            df[col] = df[col].apply(lambda x: x.split('|'))

        return df

    def __select_all_events(self):
        if self._conn is None:
            raise Exception("There is no connection")
        cur = self._conn.cursor()
        cur.execute('''
            SELECT ex."index", d1.id, d1.name, d2.id, d2.name, mechanism || ' ' || action
            FROM extraction ex
            JOIN drug d1 ON d1.name = ex.drugA
            JOIN drug d2 ON d2.name = ex.drugB
        ''')
        rows = cur.fetchall()
        headers = ["index", "id1", "name1", "id2", "name2", "event_category"]
        return pd.DataFrame(columns=headers, data=rows)

    def prep(self):
        self.load_drugs_and_events()
        if self.drugs_df is None or self.ddis_df is None:
            raise Exception("There is no data")

        drug_ids = self.drugs_df['id'].to_list()

        filtered_df = self.drugs_df
        combined_df = filtered_df.copy()

        if self.ner_df is not None and not self.ner_df.empty:
            filtered_ner_df = self.ner_df[self.ner_df['drugbank_id'].isin(
                drug_ids)]
            filtered_ner_df = self.ner_df.copy()

            # TODO: eğer kullanılan veri setinde tui, cui veya entity bilgileri yoksa o veri setine bu sütunları eklemek için aşağısı gerekli

            # idf_calc = IDF(filtered_ner_df, [f for f in filtered_ner_df.keys()])
            idf_calc = IDF(filtered_ner_df, self.ner_columns)
            idf_calc.calculate()
            idf_scores_df = idf_calc.to_dataframe()

            # for key in filtered_ner_df.keys():
            for key in self.ner_columns:
                threshold = self._ner_threshold.get(key, 0) if self._ner_threshold else 0
                # threshold = 0
                # if key.startswith('tui'):
                #     threshold = self.tui_threshold
                # if key.startswith('cui'):
                #     threshold = self.cui_threshold
                # if key.startswith('entities'):
                #     threshold = self.entities_threshold
                combined_df[key] = filtered_ner_df[key]
                valid_codes = idf_scores_df[idf_scores_df[key]
                                            > threshold].index

                # print(f'{key}: valid code size = {len(valid_codes)}')
                combined_df[key] = combined_df[key].apply(lambda items:
                                                          [item for item in items if item in valid_codes])

        moved_columns = ['id']
        moved_columns.extend(self.__similarity_related_columns__)
        chemical_properties_df = combined_df[moved_columns]

        chemical_properties_df = chemical_properties_df.fillna("").apply(list)

        # generate vectors dictionary içinde ndarray dönecek
        generated_vectors = generate_vectors(
            chemical_properties_df, self.__similarity_related_columns__)

        # TODO if necessary 
        similarity_matrices = generate_sim_matrices_new(
            chemical_properties_df, generated_vectors,  self.__similarity_related_columns__, key_column="id")

        event_categories = self.ddis_df['event_category']
        labels = event_categories.tolist()
        lb = LabelBinarizer()
        lb.fit(labels)
        classes = lb.transform(labels)

        def similarity_lambda_fnc(row, value):
            if row['id1'] in value:
                return value[row['id1']]

        def lambda_fnc(row: pd.Series, value) -> Optional[np.float16]:
            if row['id1'] in value and row['id2'] in value:
                return np.float16(np.hstack(
                    (value[row['id1']], value[row['id2']])))
            return None
            # return np.hstack(
            #     (value[row['id1']], value[row['id2']]), dtype=np.float16)

        def x_fnc(row, embeddings_after_pooling):
            if row['id1'] in embeddings_after_pooling:
                v1 = embeddings_after_pooling[row['id1']]
            else:
                v1 = np.zeros(self.embedding_size)
            if row['id2'] in embeddings_after_pooling:
                v2 = embeddings_after_pooling[row['id2']]
            else:
                v2 = np.zeros(self.embedding_size)
            return np.float16(np.hstack(
                (v1, v2)))

        for key, value in similarity_matrices.items():

            print(f'sim matrix: {key}')
            self.ddis_df[key] = self.ddis_df.apply(
                lambda_fnc, args=(value,), axis=1)
            self.columns.append(key)
            print(self.ddis_df[key].head())
        if isinstance(self, TextDatasetMixin):
            if self.embedding_dict is not None:
                for embedding_column in self.embedding_columns:
                    print(f"concat {embedding_column} embeddings")
                    embeddings_after_pooling = {k: self.pooling_strategy.apply(
                        v) for k, v in self.embedding_dict[embedding_column].items()}
                    # column_embeddings_dict = embedding_values[embedding_column]
                    self.ddis_df[embedding_column+'_embedding'] = self.ddis_df.apply(
                        x_fnc, args=(embeddings_after_pooling,), axis=1)
                    self.columns.append(embedding_column+'_embedding')

        dataframe = self.ddis_df.copy()
        if not isinstance(classes, (list, pd.Series, np.ndarray)):
            raise TypeError(
                "classes must be an iterable (list, Series, or ndarray)")

        if len(classes) != len(dataframe):
            raise ValueError(
                "Length of classes must match the number of rows in the DataFrame")

        dataframe[self.class_column] = list(classes)
        self.set_dataframe(dataframe)
