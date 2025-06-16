
from sqlite3 import Error
import sqlite3
import pandas as pd
import numpy as np


def create_connection(db_file=r"./event.db"):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn


# def select_all_drugs(conn):
#     cur = conn.cursor()
#     cur.execute(
#         '''select "index", id, name, target, enzyme, pathway, smile from drug''')
#     rows = cur.fetchall()
#     return rows


# def select_all_drugs_as_dataframe(conn):
#     headers = ['index','id', 'name', 'target', 'enzyme', 'pathway', 'smile']
#     rows = select_all_drugs(conn)
#     df = pd.DataFrame(columns=headers, data=rows)
#     df['enzyme'] = df['enzyme'].apply(lambda x: x.split('|'))
#     df['target'] = df['target'].apply(lambda x: x.split('|'))
#     df['pathway'] = df['pathway'].apply(lambda x: x.split('|'))
#     df['smile'] = df['smile'].apply(lambda x: x.split('|'))
#     return df


# def select_all_events(conn):
#     """
#     Query all rows in the event table
#     :param conn: the Connection object
#     :return:
#     """
#     cur = conn.cursor()
#     cur.execute("select * from event")

#     rows = cur.fetchall()
#     return rows


# def select_all_events_as_dataframe(conn):
#     headers = ["index", "id1", "name1", "id2", "name2", "event_category"]
#     rows = select_all_events(conn)
#     return pd.DataFrame(columns=headers, data=rows)


# def select_events_with_category(conn):
#     sql = '''select id1, name1, id2, name2, mechanism || ' ' ||action from event ev
#             join extraction ex
#             on ev.name1 = ex.drugA and ev.name2 = ex.drugB
#             union 
#             select id1, name1, id2, name2, mechanism || ' ' ||action from event ev
#             join extraction ex
#             on ev.name1 = ex.drugB and ev.name2 = ex.drugA
#             '''
#     cur = conn.cursor()
#     cur.execute(sql)

#     rows = cur.fetchall()

#     headers = ['id1', 'name1', 'id2', 'name2', 'event_category']
#     return pd.DataFrame(columns=headers, data=rows)


# def select_all_interactions_tuple_as_dataframe(conn):
#     cur = conn.cursor()
#     cur.execute("select id1, id2 from event")
#     rows = cur.fetchall()
#     headers = ['id1', 'id2']

#     return pd.DataFrame(columns=headers, data=rows)


# def select_ddi_pairs(conn):
#     cur = conn.cursor()
#     cur.execute('''
#         select d1.[index] as Drug1Index, d2.[index] as Drug2Index, 1 from event e 
#         join drug d1 on e.id1 = d1.id 
#         join drug d2 on e.id2 = d2.id 
#     ''')
#     rows = cur.fetchall()
#     return rows


# def select_ddi_pairs_as_dataframe(conn):
#     headers = ["Drug1Index", "Drug2Index", "Interaction"]
#     rows = select_ddi_pairs(conn)
#     return pd.DataFrame(columns=headers, data=rows)


# def get_interactions(conn):
#     cur = conn.cursor()
#     cur.execute('''
#         select
#             drug_1_id,
#             drug_1,
#             drug_2_id,
#             drug_2,
#             mechanism_action, 
#             interaction, 
#             masked_interaction
#         from _Interactions
#                 ''')

#     rows = cur.fetchall()

#     headers = ['id1', 'name1', 'id2', 'name2',
#                'event_category', 'interaction', 'masked_interaction']
#     df = pd.DataFrame(columns=headers, data=rows)
#     return df


# def get_extended_version(conn):
#     cur = conn.cursor()
#     cur.execute('''
#         select 
#         _Drugs."index",
#         drugbank_id,
#         _Drugs.name,
#         description,
#         synthesis_reference,
#         indication,
#         pharmacodynamics,
#         mechanism_of_action,
#         toxicity,
#         metabolism,
#         absorption,
#         half_life,
#         protein_binding,
#         route_of_elimination,
#         volume_of_distribution,
#         clearance,
#         smiles,
#         smiles_morgan_fingerprint,
#         enzymes_polypeptides,
#         targets_polypeptides
                
#         from drug 
#         join _Drugs on drug.id = _Drugs.drugbank_id  
#         where 
#                 targets_polypeptides is not null and 
#                 enzymes_polypeptides is not null and 
#                 smiles_morgan_fingerprint is not null
# ''')
#     # pathway is absent

#     rows = cur.fetchall()
#     headers = ['index', 'id', 'name', 'description', 'synthesis_reference', 'indication', 'pharmacodynamics', 'mechanism_of_action', 'toxicity', 'metabolism', 'absorption', 'half_life',
#                'protein_binding', 'route_of_elimination', 'volume_of_distribution', 'clearance', 'smiles_notation', 'smile', 'enzyme', 'target']
#     df = pd.DataFrame(columns=headers, data=rows)
#     df['smile'] = df['smile'].apply(lambda x:
#                                     np.fromstring(
#                                         x.replace(
#                                             '\n', '')
#                                         .replace('[', '')
#                                         .replace(']', '')
#                                         .replace('  ', ' '), sep=','))
#     df['enzyme'] = df['enzyme'].apply(
#         lambda x: x.split('|'))
#     df['target'] = df['target'].apply(
#         lambda x: x.split('|'))
#     return df


# SELECT
#  CASE
#  WHEN masked_interaction like '%'+drug_1+'%' THEN drug_1
#  WHEN masked_interaction like '%'+drug_2+'%' THEN drug_2
#  Else drug_2
#  END AS Absent,

# drug_1, drug_2,
# masked_interaction

# from _Interactions
# where LENGTH(masked_interaction) = LENGTH(REPLACE(masked_interaction, 'DRUG', ''))
# or LENGTH(masked_interaction) = LENGTH(REPLACE(masked_interaction, 'DRUG', '')) + 4

# if __name__ == "__main__":
#     conn = create_connection(r"./event-extended.db")
#     extended_version_df = get_extended_version(conn)

#     df = select_all_events_as_dataframe(conn)
#     print(df.head())

#     events_with_category_df = select_events_with_category(conn)
#     print(events_with_category_df.head())

#     u = events_with_category_df['event_category'].unique()
#     print(len(u))
