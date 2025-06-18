# generating reverse pairs of interactions
## DB way
INSERT INTO event
SELECT  event."index" + 37264 ,id2, name2, id1, name1, interaction
FROM event;

INSERT INTO extraction
SELECT  extraction."index" + 37264 ,mechanism, action, drugB, drugA
FROM extraction;

## Pandas way 
reverse_ddis_df = pd.DataFrame()
reverse_ddis_df['id1'] = ddis_df['id2']
reverse_ddis_df['name1'] = ddis_df['name2']
reverse_ddis_df['id2'] = ddis_df['id1']
reverse_ddis_df['name2'] = ddis_df['name1']
reverse_ddis_df['event_category'] = ddis_df['event_category']
self.ddis_df = pd.concat([ddis_df,reverse_ddis_df], ignore_index=True)