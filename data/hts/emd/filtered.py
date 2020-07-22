import data.hts.hts as hts
import data.spatial.codes as cd
import data.hts.emd.raw as rw
import data.hts.emd.cleaned as cl

YEAR = 2017
SOURCE = "codes_%d/reference_IRIS_geo%d.xls" % (YEAR, YEAR)

df_codes = cd.codes_iris(rw.path,SOURCE)
requested_communes = df_codes["commune_id"].unique()
df_households, df_persons, df_trips = cl.df_households, cl.df_persons, cl.df_trips

# Filter for non-residents
f = df_persons["commune_id"].isin(requested_communes)
df_persons = df_persons[f]

# Filter for people going outside of the area (because they have NaN distances)
remove_ids = set()

remove_ids |= set(df_trips[
    ~df_trips["origin_commune_id"].isin(requested_communes) | ~df_trips["destination_commune_id"].isin(requested_communes)
]["person_id"].unique())

remove_ids |= set(df_persons[
    ~df_persons["commune_id"].isin(requested_communes)
])

df_persons = df_persons[~df_persons["person_id"].isin(remove_ids)]

# Only keep trips and households that still have a person
df_trips = df_trips[df_trips["person_id"].isin(df_persons["person_id"].unique())]
df_households = df_households[df_households["household_id"].isin(df_persons["household_id"])]

# Finish up

df_households = df_households[hts.HOUSEHOLD_COLUMNS]
df_persons = df_persons[hts.PERSON_COLUMNS]
df_trips = df_trips[hts.TRIP_COLUMNS+["age", "sex", "employed", "studies", "has_license", "has_pt_subscription", "socioprofessional_class"]]

hts.check(df_households, df_persons, df_trips)

export_csv = df_households.to_csv(r'../../../output/df_households_MEL.csv', index=None, header=True)
export_csv2 = df_persons.to_csv(r'../../../output/df_persons_MEL.csv', index=None, header=True)
export_csv3 = df_trips.to_csv(r'../../../output/df_trips_MEL.csv', index=None, header=True)