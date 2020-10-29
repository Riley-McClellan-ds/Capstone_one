# CAN YOU STATE GOAL IN CLEAR SENTENCE???

# Backround
        Why did that get all this data
# Data Description
        What data did they collect and how did they collect it
        MENTION that because certain columns for dead trees were ALWAYS missing values when a tree was marked dead we can 
        determine that these values were likely not able to be recorded when the tree was marked dead. This may be due to the 
        format of data entry used. Hence we have replaced certain NaN values as "Dead" as they exclusively occured as NaN values when a tree was marked "dead" or "Stump" (stumps are dead it turns out).
##      Data Cleaning
        how did I clean my data

##      Data selection
        How I chose what data to use

# delete me:
<!-- If you have time make funcs that show the way you discovered that the nan types were almost entirely related to the health/status of the tree. there were only 51 exceptions to that rule all of which were likely user neglect to fill in a no problem or no issue response especially given trees nearby.  IF YOU HAVE TIME sort by LAT and LONG to determine if these values can be filled in using trees on either side -->

Notes:

Data cleaning involved the removal of one row from the dataset. This row had a NaN value for the 'health' column but 'Alive' for the status, It's trunk diameter was zero. Given these contradictions the row was removed completely. The following columns contained NaN values but still had enough useful information to retain: ['health', 'spc_latin', 'spc_common', 'steward', 'guards', 'sidewalk', 'problems']. All columns except 'health' had NaN values replaced with an "Unkown" string. While it was likely that a NaN for 'guards' or 'steward' simply meant none was observed this assumption was not made in order to reduce bias. Thankfully this represented the minority of the data 

we have NaN values in the following collumns ['health', 'spc_latin', 'spc_common', 'steward', 'guards', 'sidewalk', 'problems'], 

grouped NaN issues:
'health', 'spc_latin', 'spc_common'

grouped because these generally have Dead or Stump in status column
Will replace NaN values with string "unkown" in order to preserve information that may account for the death of the tree. Perhaps I will not use the stumps as these have been intentionally cut down and we are unable to determine the cause of death. Although I just looked it up and it's illegal to cut down street or park trees in NYC. 132 species (it is possible there are typos I must check for them)

'problems' 
If there are no problem then quite frankly that is a good thing in my mind but I need to check the values in CV dict to make sure there isn't something to account for that

'steward'
Check CV dict values

'Guards'
Check CV dict values

'sidewalk'
check CV dict values 
