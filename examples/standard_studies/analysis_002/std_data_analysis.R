library(data.table)
library(Matrix)
library(ggplot2)
library(plotly)
library(reticulate)

# to be sure we do not have stuff from
# previous R sessions
remove(list=ls())

# instructions to prepare the Python environment
use_condaenv("reticulate")
source_python("../utils/r_adapter.py")
source_python("../utils/context_funs.py")
savePythonContext()

# a list of environments to manage all the variables
# associated with different GMAP evaluation scenarios
res_env <- list()

# convenience function to add an evaluation
# to the res_env list
perform_evaluation <- function(pyfile, envname) {
  restorePythonContext()
  res_env[[envname]] <<- new.env()
  source_python(pyfile, envir=res_env[[envname]])
}

# perform the evaluations
# (we expect that every Python script produces
#  a pandas dataframe 'ref_table' and a 'datablock_list') 
perform_evaluation("gmap_fit_2017.py", "gma2017")
perform_evaluation("gmap_fit_latest.py", "gma_latest")
perform_evaluation("gmap_fit_modified.py", "gma_mod")

##################################################
#  The following functions create auxiliary
#  lists and functions that can be used to
#  convert information into human-readable form
##################################################

# create a datatable with results
for (curenv in res_env) {
  curenv[["curtable"]] <- as.data.table(curenv[["ref_table"]]) 
  curenv[["curtable"]][, c("DB_IDX", "DS_IDX") := NULL]
}

# produce a lookup table that allows to obtain for
# a dataset number a human-readable reference
expname_list <- list()
for (curenv in res_env) {
  with(curenv, {
    for (curdb in datablock_list) {
      for (curds in curdb[["datasets"]]) {
        t0 <- curds$NS
        t1 <- gsub("^ *| *$","", curds$CLABL)
        t2 <- gsub("^ *| *$","", curds$YEAR)
        expname_list[[as.character(curds$NS)]] <<- paste0(t0, " ", t1, " ", t2)
      }
    }
    remove("t0", "t1", "t2", "curdb", "curds")
  })
}
# produce a lookup table that allows to obtain for
# a reaction string a human-readable version thereof
reac_list <- list()
for (curenv in res_env) {
  with(curenv, {
    curtable[grepl("^xsid_", NODE), {
      if (is.character(DESCR[[1]])) {
        reac_list[[.BY[["REAC"]]]] <<- DESCR[[1]]
      }
      NULL
    }, by="REAC"]
  })
}

# list with human readable reaction type
mt_assoc <- list(
  "1" = "xs",
  "2" = "shape of xs",
  "3" = "ratio of xs",
  "4" = "shape of ratio of xs",
  "5" = "sum of xs",
  "6" = "SACS",
  "7" = "xs1/(xs2+xs3)",
  "8" = "shape of sum of xs",
  "9" = "shape of xs1/(xs2+xs3)",
  "10" = "ratio of SACS"
)

reac2human <- function(reac) {
  splitstrs <- strsplit(reac,'-')
  splitstrs2 <- lapply(splitstrs, strsplit, split=':')
  splitstrs3 <- lapply(splitstrs2, function(x) sapply(x, function(y) y[2]))
  splitstrs3 <- lapply(splitstrs3, as.integer)
  unlist(lapply(splitstrs3, function(x) {
    reactype <- mt_assoc[[x[1]]]
    reacs <- unlist(reac_list[x[-1]])
    if (!is.null(reactype))
      paste(reactype, paste(reacs, collapse=" - "), sep=": ")
    else
      NA_character_
  }))
}

expnum2human <- function(id) {
  res <- expname_list[as.character(id)]
  res <- lapply(res, function(x) {
    if (is.null(x)) "unknown" else x
  })
  unlist(res)
}

##################################################
#   Augment the tables with information
#   useful for plotting
##################################################

# new column with human readable versions of the reaction string
for (curenv in res_env) {
  curenv[["curtable"]][, REACSTR := reac2human(REAC)]
}
remove("curenv")

# new column with description of experimental reaction
for (curenv in res_env) {
  curenv[["curtable"]][, DESCR := NULL]
  curenv[["curtable"]][grepl("^exp_", NODE), DESCR := expnum2human(gsub("^exp_", "", NODE))]
}
remove("curenv")

# new column with fission spectrum converted to point wise representation 
tmpenv <- new.env() 
for (curenv in res_env) {
  with(tmpenv, { 
    # note: in-place modification of curenv[["curtable"]]
    curtable <- curenv[["curtable"]]
    ensfis <- curtable[NODE=='fis', ENERGY]
    valsfis <- curtable[NODE=='fis', PRIOR]
    xdiff <- diff(ensfis)
    xmid <- head(ensfis,-1) + xdiff/2
    scl_valsfis <- rep(0, length(ensfis))
    scl_valsfis[2:(length(scl_valsfis)-1)] <- valsfis[2:(length(valsfis)-1)] / diff(xmid)
    scl_valsfis[1] <- valsfis[1] / (xdiff[1]/2)
    scl_valsfis[length(scl_valsfis)] <- tail(valsfis,1) / tail(xdiff,1)/2
    valsfis <- scl_valsfis
    curtable[NODE=='fis', SCLFIS := valsfis]
  })
}
remove("tmpenv")

##################################################
#  Ok, another round of data preparation:
#  Create datatables that will be used
#  to plot the cross sections
##################################################

unique_reac <- unique(res_env[[1]][["curtable"]]$REACSTR)
plot_reac <- unique_reac[9]

min_energy <- 0.01
max_energy <- 10

# get all the evaluated cross sections
plot_evaldt_list <- lapply(res_env, function(curenv) {
  curenv[["curtable"]][
    grepl('^xsid_',NODE) & REACSTR==plot_reac & ENERGY > min_energy & ENERGY <= max_energy
  ]
})

# retrieve all the real experimental data
plot_expdt_list <- lapply(res_env, function(curenv) {
  curenv[["curtable"]][
    grepl('^exp_',NODE) & REACSTR==plot_reac & ENERGY > 0.01
    & !grepl("DUMMY", DESCR)
  ]
})

# retrieve the fission cross section for each evaluation
plot_fisdt_list <- lapply(res_env, function(curenv) {
  curenv[["curtable"]][
    NODE=='fis' & ENERGY > min_energy & ENERGY <= max_energy
  ]
}) 

# prepare a datatable that contains the predictions of
# each evaluation scenario in a distinct column (wide format)
reftable <- copy(res_env[["gma2017"]][["curtable"]])
comptable <- copy(reftable)
for (envname in names(res_env)) {
  curenv <- res_env[[envname]]
  curtable <- copy(curenv[["curtable"]])
  predname <- paste0("pred_", envname)
  keycols <- reftable[grepl("^xsid_|^norm_", NODE), list(NODE, REAC, ENERGY)]
  setkey(curtable, NODE, REAC, ENERGY)
  curpreds <- curtable[J(keycols$NODE, keycols$REAC, keycols$ENERGY), POST]
  comptable[[predname]] <- 0
  comptable[[predname]][grepl("^xsid_|^norm_", comptable$NODE)] <- curpreds
  comptable[[predname]] <- as.vector(gmapi_propagate(comptable, comptable[[predname]]))
}
remove("curtable", "keycols", "curpreds", "predname", "curenv", "envname")

##################################################
#   Ok, finally: Let's do the plotting
##################################################

# create a datatable with the sacs values for all the different scenarios
sacs_list <- lapply(res_env, function(curenv) {
  curenv[["ref_sacs"]]
})
sacs_table <- rbindlist(sacs_list, idcol="eval")

# plot the cross sections and the data on absolute scale 

unique_reacstr <- unique(comptable$REACSTR)
plot_reacstr <- unique_reacstr[9] 

ggp <- ggplot() + theme_bw()
ggp <- ggp + geom_errorbar(aes(x=ENERGY, ymin=DATA-DATAUNC, ymax=DATA+DATAUNC, col=DESCR), data=plot_expdt_list[["gma2017"]][REACSTR==plot_reacstr])
ggp <- ggp + geom_point(aes(x=ENERGY, y=DATA, col=DESCR), data=plot_expdt_list[["gma2017"]][REACSTR==plot_reacstr])

# all the evaluations
ggp <- ggp + geom_line(aes(x=ENERGY, y=POST), col="red", data=plot_evaldt_list[["gma2017"]][REACSTR==plot_reacstr])
ggp <- ggp + geom_line(aes(x=ENERGY, y=POST), col="green", data=plot_evaldt_list[["gma_latest"]][REACSTR==plot_reacstr])
ggp <- ggp + geom_line(aes(x=ENERGY, y=POST), col="blue", data=plot_evaldt_list[["gma_mod"]][REACSTR==plot_reacstr])

ggp <- ggp + ggtitle(plot_reacstr)
ggp <- ggp + scale_x_continuous(trans="log10")
print(ggp)

#ggply <- ggplotly(ggp) %>% layout(title=plot_reac)
#ggplotly(ggply)

# now plot all the data and the evaluations relative to the std 2017  

#plot_reac <- "MT:4-R1:9-R2:8"
plot_reac <- "MT:4-R1:9-R2:1"
plot_reacstr <- reac2human(plot_reac)

expmask <- comptable[,
  grepl('^exp_',NODE) & REAC==plot_reac
  & !grepl("DUMMY", DESCR)
]
  
ggp <- ggplot() + theme_bw()
ggp <- ggp + geom_line(aes(x=ENERGY, y=pred_gma_latest/pred_gma2017), data=comptable[REACSTR==plot_reacstr], col="red")
ggp <- ggp + geom_line(aes(x=ENERGY, y=pred_gma_mod/pred_gma2017), data=comptable[REACSTR==plot_reacstr], col="blue")

ggp <- ggp + geom_errorbar(aes(x=ENERGY, ymin=(DATA-DATAUNC)/pred_gma2017, ymax=(DATA+DATAUNC)/pred_gma2017, col=DESCR),
                           data=comptable[expmask & REACSTR==plot_reacstr])
ggp <- ggp + geom_point(aes(x=ENERGY, y=DATA/pred_gma2017, col=DESCR),
                           data=comptable[expmask & REACSTR==plot_reacstr])
#position=position_dodge(width=0.5))
ggp <- ggp + ylim(0.9, 1.1) + xlim(0,1e-2)
ggp <- ggp + ggtitle(plot_reacstr)
ggp <- ggp + theme(legend.position="none")
ggp
  
#ggply <- ggplotly(ggp) %>% layout(title=plot_reac)
#print(ggplotly(ggply))
