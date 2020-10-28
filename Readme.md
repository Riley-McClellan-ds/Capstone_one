# DATA

## Cleaning
Notes:

we have NaN values in the following collumns ['health', 'spc_latin', 'spc_common', 'steward', 'guards', 'sidewalk', 'problems'], 

grouped NaN issues:
'health', 'spc_latin', 'spc_common'

grouped because these generally have Dead or Stump in status column
Will replace NaN values with string "unkown" in order to preserve information that may account for the death of the tree. Perhaps I will not use the stumps as these have been intentionally cut down and we are unable to determine the cause of death. Although I just looked it up and it's illegal to cut down street or park trees in NYC

'problems' 
If there are no problem then quite frankly that is a good thing in my mind but I need to check the values in CV dict to make sure there isn't something to account for that

'steward'
Check CV dict values

'Guards'
Check CV dict values

'sidewalk'
check CV dict values 