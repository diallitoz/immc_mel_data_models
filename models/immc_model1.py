"""
File model1_swiss_metro.py
Author: DIALLO Azise Oumar
Date: 07/07/2020
Model1: first model apply to Swiss Metro
"""

##1: Librairies
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import biogeme.optimization as opt
import biogeme.messaging as msg
from biogeme.expressions import Beta,DefineVariable, bioLinearUtility

#2: Prepare the data
pd_data=pd.read_table("../data/swissmetro.dat")
print(pd_data)
#print(pd_data.describe())



#Instance of biogeme dataframe
database= db.Database("swissmetro",pd_data)
print(database.getSampleSize())

#Import the headers for future use in the formulas
#Created by the biogeme instance (headers.py), create all the colum names as variables
#So run before the creation of the instance!!!!
#from headers import *
###HEADERS BUGGGGGGG use globals()...

# The following statement allows you to use the names of the
# variable as Python variable.
globals().update(database.variables)

# Removing some observations
exclude = ((PURPOSE != 1) * (PURPOSE != 3) + (CHOICE == 0)) > 0
database.remove(exclude)
print(database.getSampleSize())

# Parameters to be estimated
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)###Keep this parameter fixed to the initial value
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)


# Definition of new variables. Method 1: using Python variables
## Easy and simple.
## But not suitable for big data and where the variables will be used more time (lot of caculation)
SM_COST = SM_CO * (GA == 0) ### SM_CO and GA come frome database columns defined as variables
TRAIN_COST = TRAIN_CO * (GA == 0)
"""
SM_COST = SM_CO * (GA == 0) ### SM_CO and GA come frome database columns defined as variables
TRAIN_COST = TRAIN_CO * (GA == 0)
CAR_AV_SP = CAR_AV * (SP != 0)
TRAIN_AV_SP = TRAIN_AV * (SP != 0)
TRAIN_TT_SCALED = TRAIN_TT / 100
TRAIN_COST_SCALED = TRAIN_COST / 100
SM_TT_SCALED = SM_TT / 100
SM_COST_SCALED = SM_COST / 100
CAR_TT_SCALED = CAR_TT / 100
CAR_CO_SCALED = CAR_CO / 100
"""

### Definition of new variables. Method 2: adding colomns to the database that can used later
# Definition of new variables by adding columns to the database.
# This is recommended for estimation. And not recommended for simulation.
CAR_AV_SP = DefineVariable('CAR_AV_SP', CAR_AV * (SP != 0), database)
TRAIN_AV_SP = DefineVariable('TRAIN_AV_SP', TRAIN_AV * (SP != 0), database)
TRAIN_TT_SCALED = DefineVariable('TRAIN_TT_SCALED', TRAIN_TT / 100.0, database)
TRAIN_COST_SCALED = DefineVariable('TRAIN_COST_SCALED', TRAIN_COST / 100, database)
SM_TT_SCALED = DefineVariable('SM_TT_SCALED', SM_TT / 100.0, database)
SM_COST_SCALED = DefineVariable('SM_COST_SCALED', SM_COST / 100, database)
CAR_TT_SCALED = DefineVariable('CAR_TT_SCALED', CAR_TT / 100, database)
CAR_CO_SCALED = DefineVariable('CAR_CO_SCALED', CAR_CO / 100, database)

# Definition of the utility functions
terms1 = [(B_TIME, TRAIN_TT_SCALED), (B_COST, TRAIN_COST_SCALED)]
V1 = ASC_TRAIN + bioLinearUtility(terms1)

terms2 = [(B_TIME, SM_TT_SCALED), (B_COST, SM_COST_SCALED)]
V2 = ASC_SM + bioLinearUtility(terms2)

terms3 = [(B_TIME, CAR_TT_SCALED), (B_COST, CAR_CO_SCALED)]
V3 = ASC_CAR + bioLinearUtility(terms3)

"""
V1 = ASC_TRAIN + \
     B_TIME * TRAIN_TT_SCALED + \
     B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + \
     B_TIME * SM_TT_SCALED + \
     B_COST * SM_COST_SCALED
V3 = ASC_CAR + \
     B_TIME * CAR_TT_SCALED + \
     B_COST * CAR_CO_SCALED
"""
# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}

# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = models.loglogit(V, av, CHOICE)

# Define level of verbosity
logger = msg.bioMessage()
#logger.setSilent()
#logger.setWarning()
logger.setGeneral()
#logger.setDetailed()


# These notes will be included as such in the report file.
userNotes = ('Example of a logit model with three alternatives: Train, Car and Swissmetro.'
             ' Same as 01logit, using bioLinearUtility, and introducing some options '
             'and features.')

# Create the Biogeme object
biogeme = bio.BIOGEME(database, logprob, numberOfThreads=2, userNotes=userNotes)
biogeme.modelName = '01logitBis'

# Estimate the parameters
results = biogeme.estimate(bootstrap=100,
                           algorithm=opt.bioNewton,
                           algoParameters={'hamabs': True},
                           saveIterations=True)

biogeme.createLogFile(verbosity=3)

# Get the results in a pandas table
print('Parameters')
print('----------')
pandasResults = results.getEstimatedParameters()
print(pandasResults)

# Get general statistics
print('General statistics')
print('------------------')
stats = results.getGeneralStatistics()
for description, (value, formatting) in stats.items():
    print(f'{description}: {value:{formatting}}')

# Messages from the optimization algorithm
print('Optimization algorithm')
print('----------------------')
for description, message in results.data.optimizationMessages.items():
    print(f'{description}:\t{message}')
