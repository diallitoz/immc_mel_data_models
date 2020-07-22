import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


import numpy as np
import data.hts.emd.raw as raw
import data.hts.hts as hts


PURPOSE_MAP = [
    ("1", "home"),
    ("2", "home"),
    ("11", "work"),
    ("12", "work"),
    ("22", "education"),
    ("23", "education"),
    ("24", "education"),
    ("25", "education"),
    ("26", "education"),
    ("27", "education"),
    ("28", "education"),
    ("29", "education"),
    ("30", "shop"),
    ("31", "shop"),
    ("32", "shop"),
    ("33", "shop"),
    ("34", "shop"),
    ("35", "other"),
    ("41", "other"),
    ("42", "other"),
    ("43", "other"),
    ("51", "leisure"),
    ("52", "leisure"),
    ("53", "leisure"),
    ("54", "leisure"),
    ("61", "other"),
    ("62", "other"),
    ("63", "other"),
    ("64", "other"),
    ("71", "other"),
    ("72", "other"),
    ("73", "other"),
    ("74", "other"),
    ("81", "other"),
    ("82", "other"),
    ("91", "other")
]

MODES_MAP = [
    ("1", "walk"),
    ("10", "bike"), # bike free floting
    ("11", "bike"), # bike
    ("12", "bike"), # bike
    ("13", "car"), # 2 wheeled
    ("14", "car_passenger"), # motorcycle passenger
    ("15", "car"), # 2 wheeled >50cm
    ("16", "car_passenger"), # motorcycle passenger
    ("21", "car"),
    ("22", "car_passenger"),
    ("31", "pt"), # bus
    ("32", "pt"), # tram
    ("33", "pt"), # metro
    ("34", "pt"), #Lijn
    ("35", "pt"), # TEC
    ("39", "pt"),  # autre reseau urbain
    ("41", "pt"),  # Arc-en-ciel
    ("51", "pt"),  # train
    ("61", "pt"),  # taxi
    ("71", "pt"),  # transport employeur
    ("81", "car"), # Conducteur Camion, fourgon
    ("82", "car_passenger"),# passager Conducteur Camion, fourgon
    ("91","pt"), # Fluvial ou maritime
    ("92", "pt"), # Plane
    ("93", "bike"),  # roller
    ("94", "bike"),  # fauteuil roulant
    ("95", "bike"),  # skate
    ("96", "bike"),  # trottinette
    #("97","pt"), #Autres
]

MODES_MECA_MAP = [
("['pt']","pt"),
("['car']", "car"),
("['car_passenger']", "car_passenger"),
("['pt', 'pt']", "pt+pt"),
("['car_passenger', 'pt', 'pt']", "pt+car_passenger"),
("['pt', 'car']", "pt+car"),
("['bike', 'pt', 'bike']", "pt+bike"),
("['bike']", "bike"),
("['pt', 'pt', 'car_passenger']", "pt+car_passenger"),
("['car_passenger', 'pt']", "pt+car_passenger"),
("['car', 'car_passenger']", "car+car_passenger"),
("['car_passenger', 'car']", "car+car_passenger"),
("['pt', 'car_passenger']", "pt+car_passenger"),
("['pt', 'pt', 'pt']", "pt+pt"),
("['car', 'pt', 'pt']", "pt+car"),
("['pt', 'pt', 'pt', 'pt']" , "pt+pt"),
("['pt', 'pt', 'car']", "pt+car"),
("['bike', 'car']", "car+bike"),
("['bike', 'pt', 'pt', 'pt']", "pt+bike"),
("['pt', 'pt', 'pt', 'bike']", "pt+bike"),
("['car_passenger', 'pt', 'car']", "pt+car+car_passenger"),
("['bike', 'pt', 'pt']", "pt+bike"),
("['pt', 'bike']", "pt+bike"),
("['car_passenger', 'car_passenger']", "car_passenger"),
("['car', 'pt']", "pt+car"),
("['bike', 'pt']", "pt+bike"),
("['car', 'pt', 'car']", "pt+car"),
("['bike', 'pt', 'pt', 'bike']", "pt+bike"),
("['pt', 'pt', 'pt', 'car_passenger']", "pt+car_passenger"),
("['car', 'car']", "car"),
("['pt', 'pt', 'bike']", "pt+bike"),
("['car_passenger', 'pt', 'pt', 'car']", "pt+car+car_passenger"),
("['car_passenger', 'pt', 'pt', 'pt']", "pt+car_passenger"),
("['car', 'pt', 'pt', 'pt', 'pt']", "pt+car"),
("['car_passenger', 'pt', 'car_passenger']", "pt+car_passenger"),
("['car_passenger', 'bike']", "car_passenger+bike"),
("['car_passenger', 'pt', 'pt', 'car_passenger']", "pt+car_passenger"),
("['car_passenger', 'car_passenger', 'pt', 'pt']", "pt+car_passenger"),
("['car', 'pt', 'pt', 'pt']", "pt+car"),
("['pt', 'pt', 'pt', 'car']", "pt+car"),
("['car', 'bike']", "car+bike"),
("['pt', 'pt', 'car_passenger', 'car']", "pt+car+car_passenger")
]

VOT_TRIP_PURPOSE_MAP = [
    ("home", 10.0),
    ("work", 17.5),
    ("education", 10.0),
    ("shop", 6.8),
    ("leisure", 6.8),
    ("other", 7.9)
]

VOT_MODE_MAP = [
    ("car", 10.0),
    ("car_passenger", 17.5),
    ("pt", 10.0),
    ("bike", 6.8),
    ("walk", 6.8)
]

df_households, df_persons, df_trips, df_trajets = raw.df_menages,raw.df_personnes,raw.df_deplacements,raw.df_trajets

# Make copies
df_households = pd.DataFrame(df_households, copy = True)
df_persons = pd.DataFrame(df_persons, copy = True)
df_trips = pd.DataFrame(df_trips, copy = True)

# Construct new IDs for households, persons and trips (which are unique globally)
df_households["menage_id_constr"] = df_households["ZFM"] + df_households["ECH"]
df_households["menage_id_constr"] = df_households["menage_id_constr"].astype(np.int)
df_households["household_id"] = np.arange(len(df_households))

df_persons["menage_id_constr"] = df_persons["ZFP"] + df_persons["ECH"]
df_persons["menage_id_constr"] = df_persons["menage_id_constr"].astype(np.int)
df_persons = pd.merge(
    df_persons, df_households[["menage_id_constr", "household_id"]],
    on="menage_id_constr"
)
df_persons["person_id_constr"] = df_persons["ZFP"] + df_persons["ECH"] + df_persons["PER"]
df_persons["person_id_constr"] = df_persons["person_id_constr"].astype(np.int)
df_persons["person_id"] = np.arange(len(df_persons))

# df_persons = df_persons.drop_duplicates("person_id_constr")###Suppression des ID construits qui sont identiques
df_persons["person_id"] = np.arange(len(df_persons))
# print("Personne après après\n", df_persons)

df_trips["menage_id_constr"] = df_trips["ZFD"] + df_trips["ECH"]
df_trips["person_id_constr"] = df_trips["ZFD"] + df_trips["ECH"] + df_trips["PER"]
df_trips["menage_id_constr"] = df_trips["menage_id_constr"].astype(np.int)
df_trips["person_id_constr"] = df_trips["person_id_constr"].astype(np.int)
df_trips = pd.merge(
    df_trips, df_persons[["menage_id_constr", "person_id_constr", "household_id", "person_id"]],
    on=["menage_id_constr", "person_id_constr"]
)
df_trips["deplacement_id_constr"] = df_trips["ZFD"] + df_trips["ECH"] + df_trips["PER"] + df_trips["NDEP"]
df_trips["deplacement_id_constr"] = df_trips["deplacement_id_constr"].astype(np.int)
df_trips["trip_id"] = np.arange(len(df_trips))

df_trajets["menage_id_constr"] = df_trajets["ZFT"] + df_trajets["ECH"]
df_trajets["person_id_constr"] = df_trajets["ZFT"] + df_trajets["ECH"] + df_trajets["PER"]
df_trajets["deplacement_id_constr"] = df_trajets["ZFT"] + df_trajets["ECH"] + df_trajets["PER"] + df_trajets["NDEP"]
df_trajets["menage_id_constr"] = df_trajets["menage_id_constr"].astype(np.int)
df_trajets["person_id_constr"] = df_trajets["person_id_constr"].astype(np.int)
df_trajets["deplacement_id_constr"] = df_trajets["deplacement_id_constr"].astype(np.int)
df_trajets = pd.merge(df_trajets, df_trips[["menage_id_constr", "person_id_constr", "deplacement_id_constr",
                                            "household_id", "person_id", "trip_id"]],
                      on=["menage_id_constr", "person_id_constr", "deplacement_id_constr"]
                      )
df_trajets["trajet_id_constr"] = df_trajets["ZFT"] + df_trajets["ECH"] + df_trajets["PER"] + df_trajets["NDEP"] + \
                                 df_trajets["T1"]
df_trajets["trajet_id_constr"] = df_trajets["trajet_id_constr"].astype(np.int)
df_trajets["trajet_id"] = np.arange(len(df_trajets))


# Weight
df_persons["person_weight"] = df_persons["COE1"].astype(np.float)
df_households["household_weight"] = df_households["COE0"].astype(np.float)

# Clean age
df_persons["age"] = df_persons["P4"].astype(np.int)

# Clean sex
df_persons.loc[df_persons["P2"] == 1, "sex"] = "male"
df_persons.loc[df_persons["P2"] == 2, "sex"] = "female"
df_persons["sex"] = df_persons["sex"].astype("category")

# Household size
# There is not this attribute in df_household. There we need to construct it!
df_households["NP"] = 100
households_id = df_persons["household_id"]
for household_id in list(set(households_id)):
    # get subset from df where menage_id_constr = id_menage:
    df_household = df_persons[df_persons["household_id"] == household_id]
    df_households.loc[df_households["household_id"] == household_id, "NP"] = len(df_household)
df_households["household_size"] = df_households["NP"].astype(np.int)

# Clean departement (MEL is in departement 59) even this feature is not really relevant for this study
df_persons["departement_id"] = 59
df_households["departement_id"] = 59
df_trips["origin_departement_id"] = 59
df_trips["destination_departement_id"] = 59

# Clean zone fine
df_persons["zone_fine_id"] = df_persons["ZFP"].astype(np.str).astype("category")
df_persons["work_study_zone_fine_id"] = df_persons["P15"].fillna("aaaaa").astype(np.str).astype \
    ("category"  )##Lieu de travail ou d'études
df_households["zone_fine_id"] = df_households["ZFM"].astype(np.str).astype("category")
df_trips["origin_zone_fine_id"] = df_trips["D3"].astype(np.str).astype("category")
df_trips["destination_zone_fine_id"] = df_trips["D7"].astype(np.str).astype("category")

# Clean commune
df_persons["commune_id"] = df_persons["IDP4"].astype(np.str).astype("category")
df_households["commune_id"] = df_households["IDM4"].astype(np.str).astype("category")
df_trips["commune_id"] = df_trips["IDD4"].astype(np.str).astype("category")
df_trips["origin_commune_id"] = df_trips["GDO1"].fillna("aaaaa").astype(np.str).astype \
    ("category"  )  # Insee Zone fine Origine
df_trips["destination_commune_id"] = df_trips["GDD1"].fillna("aaaaa").astype(np.str).astype \
    ("category"  )  # Insee Zone fine Destination

###Drop trips from aaaaa or to aaaaa

#df_trips.drop([df_trips["origin_commune_id"]=="aaaaa"], inplace=True)
#df_trips.drop([df_trips["destination_commune_id"]=="aaaaa"], inplace=True)
#df_trips = df_trips[~df_trips["origin_commune_id"].isin(["aaaaa"]) | ~df_trips["destination_commune_id"].isin(["aaaaa"])]
df_trips = df_trips[~df_trips["origin_commune_id"].isin(["aaaaa"])]
df_trips = df_trips[~df_trips["destination_commune_id"].isin(["aaaaa"])]

# Clean employment
df_persons["employed"] = df_persons["P9"].isin([1, 2, 3]  )##1: temps plein,   2: temps partiel et 3: apprentit,
# stage, formation

# Studies
df_persons["studies"] = df_persons["P9"].isin([4, 5]  )##4:etudiant  5: scolaire jusqu'au BAC

# Calculate consumption units
hts.check_household_size(df_households, df_persons)
df_households = pd.merge(df_households, hts.calculate_consumption_units(df_persons), on = "household_id")

# Socioprofessional class
df_persons["socioprofessional_class"] = df_persons["PCSD"].fillna(00).astype(int)

# Number of vehicles
df_households["number_of_vehicles"] = df_households["M6"] + df_households["M14"  ]##M6: Voiture particulière
# M14: Nbre de 2/3 roues motorisées
df_households["number_of_vehicles"] = df_households["number_of_vehicles"].astype(np.int)
df_households["number_of_bikes"] = df_households["M21"].astype(np.int)

# License
df_persons["has_license"] = df_persons["P7"] == 1

# Car availability
df_persons = pd.merge(
    df_persons, df_households[["household_id", "number_of_vehicles", "number_of_bikes"]],
    on="household_id"
)

df_persons["car_availability"] = df_persons["number_of_vehicles"]>0 & df_persons["has_license"]

# Bike availability
df_persons["bike_availability"] = df_persons["number_of_bikes"]>0

# Has subscription
df_persons["has_pt_subscription"] = df_persons["P12"].isin([1, 2, 3, 5, 6])##1: oui et gratuit, 2: Oui et payant avec
# partie employeur, 3: oui et payant entierement à votre charge, 5: Oui, payant (sans information sur la prise
# en charge), 6: Oui, mais sans précision

# Household income
## Non Disponible pour l'instant. Solution : faire un matchning avec les données de recensement
# df_households["income_class"] = df_households["REVENU"] - 1
# df_households.loc[df_households["income_class"].isin([10.0, 11.0, np.nan]), "income_class"] = -1
# df_households["income_class"] = df_households["income_class"].astype(np.int)
df_households["income_class"] = 0
df_households["income_class"] = df_households["income_class"].astype(np.int)

# Trip purpose
df_trips["following_purpose"] = "other"
df_trips["preceding_purpose"] = "other"

for prefix, activity_type in PURPOSE_MAP:
    df_trips.loc[
        df_trips["D5A"].astype(np.str).str.startswith(prefix), "following_purpose"##D5A : Motif destination du
        # deplacement
    ] = activity_type

    df_trips.loc[
        df_trips["D2A"].astype(np.str).str.startswith(prefix), "preceding_purpose"##D2A : Motif origine du
        # deplacement considéré comme le motif du deplacement precedent
    ] = activity_type

df_trips["following_purpose"] = df_trips["following_purpose"].astype("category")
df_trips["preceding_purpose"] = df_trips["preceding_purpose"].astype("category")

# Trip mode
df_trips["mode"] = "pt"

for prefix, mode in MODES_MAP:
    df_trips.loc[
        df_trips["MODP"].astype(np.str).str.startswith(prefix), "mode"
    ] = mode

###To do, take the real mode motorized from TRAJETS Table
df_trips["mode"] = df_trips["mode"].astype("category")

##Trips performed by walking
df_trips_walk=df_trips[df_trips["mode"]=="walk"]

# Trajets mode
df_trajets["modeM"] = "pt"

for prefix, mode in MODES_MAP:
    df_trajets.loc[
        df_trajets["T3"].astype(np.str).str.startswith(prefix), "modeM"
    ] = mode
df_trajets["modeM"] = df_trajets["modeM"].astype("category")

##AccesEgress time for TRAJET
df_trajets["accessEgressTime"] = df_trajets["T2"] + df_trajets["T6"]

### Regroupement des trajets selon le deplacement
df_trajets = df_trajets.groupby("trip_id")[["GT1", "T2", "T3", "modeM", "T4", "GTO1", "T5", "GTD1", "T6", "accessEgressTime"]].agg(lambda x: (x.tolist()))
df_trajets.reset_index(inplace= True)

## Accessement of intermodal trips TRAJETS
for prefix, mode in MODES_MECA_MAP:
    df_trajets.loc[
        df_trajets["modeM"].astype(np.str).str.startswith(prefix), "modeM"
    ] = mode

df_trips = pd.merge(
        df_trips, df_trajets[["trip_id", "modeM", "accessEgressTime"]],
        on=["trip_id"]
    )

#Adding walking trips
df_trips = pd.concat([df_trips, df_trips_walk],ignore_index=True,sort=False)

## Fixing NaN elements
df_trips["modeM"] = df_trips["modeM"].fillna("walk")
df_trips["accessEgressTime"] = df_trips["accessEgressTime"].fillna(0)

# Trip flags
df_trips = hts.compute_first_last(df_trips)

# Further trip attributes
df_trips["routed_distance"] = df_trips["D12"] ##Distance parcourue en metres
df_trips["routed_distance"] = df_trips["routed_distance"].fillna(0.0)
df_trips["euclidean_distance"] = df_trips["D11"  ]## longeur à Vol d'oiseau en metres

# Trip times
df_trips["departure_time"] = df_trips["D4"].str[:2].astype(np.int) * 3600 + df_trips["D4"].str[2:4].astype(np.int) * 60
# heure de depart en HHMM -> convertie en sec
df_trips["arrival_time"] = df_trips["D8"].str[:2].astype(np.int) * 3600 + df_trips["D8"].str[2:4].astype(np.int) * 60
df_trips = hts.fix_trip_times(df_trips)

# Durations
df_trips["trip_duration"] = df_trips["arrival_time"] - df_trips["departure_time"]
hts.compute_activity_duration(df_trips)

# Add some person's social-demo characteristics to trips
df_trips = pd.merge(
    df_trips, df_persons[["person_id", "age", "sex", "employed", "studies", "has_license", "has_pt_subscription", "socioprofessional_class"]], on = "person_id", how = "left"
)

# Cost
## Method to compute cost of vehicule car and pt considering duration, distance....
#Nombre de kilomètres à parcourir, présence de péages, consommation du véhicule, coût du carburant au litre, usure du véhicule
#En règle générale, on considère qu’un véhicule de tourisme moyen consomme 0,07 litre par km selon les chiffres de l’Ademe
##Prix du carburant depend de la motorisation (diesel ou essence) mais aussi selon le carburant que vous utilisez : sans plomb 95-E10,
# sans plomb 95, sans plomb 98, superéthanol E85
# On decide de prendre un prix moyen de
# Formuile du trajet = nombre de kilomètres x consommation du véhicule au kilomètre x prix du carburant au kilomètre
## TO DO: Prise en compte des peages et des types de PT (bus, metro, tram vs train, TGV...)
def vehicule_cost(df_trips, conso, fuel_price=0.07,pt_ticket_price=1.5):
    df_trips["cost"] = 0.0
    df_trips.loc[df_trips["modeM"] == "car", "cost"] = df_trips["routed_distance"] * 1e-3 * conso * fuel_price##Non prise en compte du stationnement et des frais de peage eventuels

    no_pt_sub = df_trips["has_pt_subscription"] == False

    df_trips.loc[df_trips["modeM"] == "pt", "cost"] = pt_ticket_price * no_pt_sub

    df_trips.loc[df_trips["modeM"] == "pt+pt", "cost"] = pt_ticket_price * no_pt_sub#Non prise en compte des trains et autres PT lourds

    df_trips.loc[df_trips["modeM"] == "pt+car", "cost"] = pt_ticket_price * no_pt_sub + df_trips["routed_distance"] * 1e-3 * conso * fuel_price * 0.5

    df_trips.loc[df_trips["modeM"] == "pt+car_passenger", "cost"] = pt_ticket_price * no_pt_sub + df_trips["routed_distance"] * 1e-3 * conso * fuel_price * 0.5

    df_trips.loc[df_trips["modeM"] == "pt+bike", "cost"] = pt_ticket_price * no_pt_sub


    #df_trips.loc[df_trips["modeM"] == "pt+car" & df_trips["has_pt_subscription"]==False, "cost"] = pt_ticket_price + df_trips["routed_distance"] * 1e-3 * 0.5 * conso * fuel_price

    #df_trips.loc[df_trips["modeM"] == "pt+car_passenger" & df_trips["has_pt_subscription"]==False, "cost"] = pt_ticket_price

    #df_trips.loc[df_trips["modeM"] == "ppt+bike" & df_trips["has_pt_subscription"]==False, "cost"] = pt_ticket_price


    return df_trips

vehicule_cost(df_trips,0.07,1.52,1.5)


# Value of time (VoT)
vot_urbain = df_trips["origin_commune_id"].astype(np.int)==df_trips["destination_commune_id"].astype(np.int)
vot_inter_urbain = df_trips["origin_commune_id"].astype(np.int)!=df_trips["destination_commune_id"].astype(np.int)
df_trips["vot"] = 7.9
for prefix, vot in VOT_TRIP_PURPOSE_MAP:
    df_trips.loc[df_trips["following_purpose"].astype(np.str).str.startswith(prefix) & vot_urbain, "vot"] = vot
df_trips["vot"] = df_trips["vot"].astype(np.float)

# Add weight to trips
df_trips = pd.merge(
    df_trips, df_persons[["person_id", "person_weight"]], on = "person_id", how = "left"
).rename(columns = { "person_weight": "trip_weight" })
df_persons["trip_weight"] = df_persons["person_weight"]

# Chain length
# There is not this attribute in df_persons. There we need to construct it!
df_persons["number_of_trips"] = 0
ID_person = df_trips["person_id"]
for id_person in list(set(ID_person)):
    # get subset from df where person_id = id_person:
    df_person = df_trips[df_trips["person_id"] == id_person]
    df_persons.loc[df_persons["person_id"] == id_person, "number_of_trips"] = len(df_person)
df_persons["number_of_trips"] = df_persons["number_of_trips"].astype(np.int)

# Passenger attribute
df_persons["is_passenger"] = df_persons["person_id"].isin(
    df_trips[df_trips["mode"] == "car_passenger"]["person_id"].unique()
)



# Drop people that have NaN departure or arrival times in trips
# Filter for people with NaN departure or arrival times in trips
f = df_trips["departure_time"].isna()
f |= df_trips["arrival_time"].isna()

f = df_persons["person_id"].isin(df_trips[f]["person_id"])

nan_count = np.count_nonzero(f)
total_count = len(df_persons)

print("Dropping %d/%d persons because of NaN values in departure and arrival times" % (nan_count, total_count))

df_persons = df_persons[~f]
df_trips = df_trips[df_trips["person_id"].isin(df_persons["person_id"].unique())]
df_households = df_households[df_households["household_id"].isin(df_persons["household_id"])]
