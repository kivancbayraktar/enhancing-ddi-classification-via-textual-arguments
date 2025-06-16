from ner.ner import CTakesNER
from ddi_fw.datasets.db_utils import create_connection

import pandas as pd
import numpy as np
from datasets import IDF


def __select_all_drugs_as_dataframe__(conn):

    cur = conn.cursor()
    cur.execute(f'''
            select
            _Drugs."index",
            drugbank_id,
            _Drugs.name,
            description,
            synthesis_reference,
            indication,
            pharmacodynamics,
            mechanism_of_action,
            toxicity,
            metabolism,
            absorption,
            half_life,
            protein_binding,
            route_of_elimination,
            volume_of_distribution,
            clearance,
            smiles,
            smiles_morgan_fingerprint,
            enzymes_polypeptides,
            targets_polypeptides,
            pathways,
            tuis_description,
            cuis_description,
            entities_description

            from _Drugs


    ''')
    #  where
    #         targets_polypeptides is not null and
    #         enzymes_polypeptides is not null and
    #         pathways is not null and
    #         smiles_morgan_fingerprint is not null
    #        drugbank_id in {format(param)} and

    # pathway is absent

    rows = cur.fetchall()
    headers = ['index', 'drugbank_id', 'name', 'description', 'synthesis_reference', 'indication', 'pharmacodynamics', 'mechanism_of_action', 'toxicity', 'metabolism', 'absorption', 'half_life',
               'protein_binding', 'route_of_elimination', 'volume_of_distribution', 'clearance', 'smiles_notation', 'smile', 'enzyme', 'target', 'pathway',
               'tui_description', 'cui_description', 'entities_description']
    df = pd.DataFrame(columns=headers, data=rows)
    df['smile'] = df['smile'].apply(lambda x:
                                    np.fromstring(
                                        x.replace(
                                            '\n', '')
                                        .replace('[', '')
                                        .replace(']', '')
                                        .replace('  ', ' '), sep=','))
    df['enzyme'] = df['enzyme'].apply(
        lambda x: x.split('|') if x is not None else [])
    df['target'] = df['target'].apply(
        lambda x: x.split('|') if x is not None else [])
    df['pathway'] = df['pathway'].apply(
        lambda x: x.split('|') if x is not None else [])
    df['tui_description'] = df['tui_description'].apply(
        lambda x: x.split('|') if x is not None else [])
    df['cui_description'] = df['cui_description'].apply(
        lambda x: x.split('|') if x is not None else [])
    df['entities_description'] = df['entities_description'].apply(
        lambda x: x.split('|') if x is not None else [])
    return df


db = f'data/drugbank/db/zips/drugbank.db'
conn = create_connection(db)
drugs_df = __select_all_drugs_as_dataframe__(conn)


ctakes_api_url = 'http://localhost:8080/analyze?pipeline=Default'
ctakes_ner = CTakesNER(
    drugs_df=drugs_df,
    api_url=ctakes_api_url,
    output_path='ner/output/ctakes',
    columns=['description', 'indication'
             , 'synthesis_reference', 'pharmacodynamics',
             'mechanism_of_action', 'toxicity', 'metabolism', 'absorption', 'half_life', 'protein_binding', 'route_of_elimination','volume_of_distribution', 'clearance'],
    ids=[])
# ctakes_ner.run(run_for=['route_of_elimination'])
# override columns
# ctakes_ner.run(run_for=['description', 'indication', 'synthesis_reference', 'pharmacodynamics',
#              'mechanism_of_action', 'toxicity', 'metabolism', 'absorption', 'half_life', 'protein_binding', 'volume_of_distribution', 'clearance'])
# ner_df = ctakes_ner.create_dataframe(override=True)
# print(ner_df.keys())
# print(ner_df.shape)
# print(ner_df.head())
# print(ner_df.head())

loaded_ner_df = ctakes_ner.load('ner/output/ctakes/ctakes_ner.pkl')
# print(loaded_ner_df.tail)
# print(loaded_ner_df.shape)
# print(loaded_ner_df[loaded_ner_df['drugbank_id'] == 'DB17386']['entities_clearance'])