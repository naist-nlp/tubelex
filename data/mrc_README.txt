The mrc.csv file is extracted from the machine-readable MRC database availabe here:

https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/1054

Citation: Coltheart, M. (Max), 1939- and Wilson, Michael John, 1939-, 1987, MRC Psycholinguistic Database Machine Usable Dictionary : expanded Shorter Oxford English Dictionary entries / Max Coltheart and Michael Wilson, Oxford Text Archive, http://hdl.handle.net/20.500.12024/1054.

License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
http://creativecommons.org/licenses/by-nc-sa/3.0/

Words with available (non-zero) psycholinguistic variables were extracted using the following steps:

1. Compile the `dict` command (dict.c).

2. Run the following commands to create mrc.csv

echo 'WORD,FAM,CONC,IMAG,MEANC,MEANP,AOA' > mrc.csv                                                                                                       
./dict -W -I -J -K -L -M -N |  grep -v '^ 0  0  0  0  0  0  \|^ 321.*WITNESS' | sort -uk7 | tr A-Z a-z | awk 'BEGIN{OFS=","} {print $7,$1?$1/100:"",$2?$2/100:"",$3?$3/100:"",$4?$4/100:"",$5?$5/100:"",$6?$6/100:""}' >> mrc.csv

Note that the word witness apears twice in the data. We take the record that has both more available data (as well as higher values). The resulting file contains only unique words. We transform the values to the original decimal scale (by dividing them with 100), and replace zeros with N/A (empty fields) for consistency.