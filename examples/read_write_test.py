from database_reading import read_gma_database
from database_writing_utils import extract_prior_datatable, write_prior_text
from database_writing_utils import extract_experiment_datatable
from database_writing_utils import (extract_fission_datatable, write_fission_text,
        write_mode_setup, write_IPP_setup, write_control_header,
        extract_dataset_from_datablock, generate_dataset_text)
from GMAP import run_GMA_program
from os.path import join

orig_dbfile = 'tests/test_002/input/data.gma'
outdir = 'tmptest'

gma_database = read_gma_database(orig_dbfile)
datablock_list = gma_database['datablock_list']

prior_dt = extract_prior_datatable(gma_database['APR'])
fission_dt = extract_fission_datatable(gma_database['fisdata'])
expdata_dt = extract_experiment_datatable(datablock_list, prior_dt)

MODC = datablock_list[0].MODC
MOD2 = datablock_list[0].MOD2
MPPP = gma_database['MPPP']
MODAP = gma_database['MODAP']
IPP = gma_database['IPP']

gma_database.keys()

format_dic = {'format109': "(2E10.4,12F8.4)"}


text = []
mode_text = write_mode_setup(MODC, MOD2, 0, MODAP, MPPP)
ipp_text = write_IPP_setup(IPP[1], IPP[2], IPP[3], IPP[4], IPP[5], IPP[6], IPP[7], IPP[8])
fis_text = write_fission_text(fission_dt)
prior_text = write_prior_text(prior_dt)

# write the datasets
datablock_texts = []
for datablock in datablock_list:
    newline = write_control_header('BLCK', 0, 0, 0, 0, 0, 0, 0, 0)
    datablock_texts.append(newline)
    for id in range(1, datablock.num_datasets+1):
        dataset = extract_dataset_from_datablock(id, datablock)
        dataset_text = generate_dataset_text(dataset, format_dic=format_dic) 
        datablock_texts.append(dataset_text)
    newline = write_control_header('EDBL', 0, 0, 0, 0, 0, 0, 0, 0)
    datablock_texts.append(newline)

datablocks_text = '\n'.join(datablock_texts)
eof_marker = write_control_header('END*', 0, 0, 0, 0, 0, 0, 0, 0)
     
file_content = '\n'.join([
    mode_text, ipp_text, fis_text, prior_text, datablocks_text, eof_marker 
    ])


with open(join(outdir,'data2.gma'), 'w') as f:
    f.write(file_content)

run_GMA_program(dbfile=orig_dbfile, resfile=join(outdir,'gma1.res'),
        plotfile=join(outdir,'plot1.dta'))

run_GMA_program(dbfile=join(outdir,'data2.gma'), resfile='tmptest/gma2.res', plotfile='tmptest/plot2.dta',
        format_dic=format_dic)


