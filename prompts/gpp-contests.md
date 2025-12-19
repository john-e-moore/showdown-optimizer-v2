**Objective**
For pipeline A, we currently have --gpp-category being read dynamically. I don't like this -- I want to store all GPP contest types in a file and keep them static.

**Details**
- I still want to be able to specify a --gpp-category flag.
- Bins:
 - Sport (e.g. NBA, NFL)
 - Contest type (e.g. Classic, Showdown)
 - Size (0-1k, 1k-10k, 10k+)
 - Num Entries - for now, let's categorize anything with 1-5 entries max as "single entry" and anything more as "mme". We will refine this down the road.
- So, create a bin for each combination of the 4 above parameters.
- Be sure to update the README with info about the bins and how to run the pipeline on a specific one.