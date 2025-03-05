#!/usr/bin/env python
# coding: utf-8

import glob
import pandas as pd
import numpy as np
import awkward as ak
from itertools import cycle
from pathlib import Path

import utils.tools as tools
import utils.plotting as plotting

import mplhep as cms
import matplotlib.pyplot as plt
cms.style.use("CMS")
plt.rcParams["figure.figsize"] = (10,7)


# input data definition
# put "default" objects first
# i.e. those that should be used to obtain fixed rate

nComp = 4

l1Labels = ['Default', 'Default_noPUM', 'BaselineZS', 'ConservativeZS']
branchTypes = ['unp', 'emu', 'emu', 'emu'] # unp or emu

rootDir = "/eos/cms/store/group/dpg_trigger/comm_trigger/L1Trigger/bundocka/metzs/"

sigPaths  = ["zmu_base/", "zmu_pumOff/", "zmu_base/", "zmu_con/"]
bkgPaths  = ["zb_base/", "zb_pumOff/", "zb_base/", "zb_con/"]


inputFormat = 'nano'     # nanoAOD
#inputFormat = 'hdf5'     # pandas dataframes

sigName = "zmu"
bkgName = "zb"

writeDir = "./data/"

for label in l1Labels:
       Path(writeDir+"/"+label).mkdir(parents=True, exist_ok=True)

fileName = "nano_1*.root"

sigFilesArr = [glob.glob(rootDir + path + fileName) for path in sigPaths]
bkgFilesArr = [glob.glob(rootDir + path + fileName) for path in bkgPaths]

if len(l1Labels) != nComp or len(branchTypes) != nComp or len(sigFilesArr) != nComp or len(bkgFilesArr) != nComp:
       raise TypeError("Number of inputs datasets is not consistent")

# L1 thresholds (GeV)
l1JetThresholds = [30, 120, 180]
l1METThresholds = [50, 90]

# arrays containing our signal and background data
# for the different sets of input files
sigs = []
bkgs = []

sig_dfs = []
bkg_dfs = []


if inputFormat == 'nano':
    
    for sigFiles, branchType, l1Label in  zip(sigFilesArr, branchTypes, l1Labels):
        nFiles = len(sigFiles)
        tools.getDataframes(sigFiles[:nFiles], tools.getBranches(['Jet'], branchType=='emu', False), l1Label, writeDir, True)
    for bkgFiles, branchType, l1Label in  zip(bkgFilesArr, branchTypes, l1Labels):
        nFiles = len(bkgFiles)
        tools.getDataframes(bkgFiles[:nFiles], tools.getBranches(['Jet'], branchType=='emu', False), l1Label, writeDir, False)

sig_hdf5FilesArr = [glob.glob(writeDir + "/" + label + "/*sig.hdf5") for label in l1Labels]
bkg_hdf5FilesArr = [glob.glob(writeDir + "/" + label + "/*bkg.hdf5") for label in l1Labels]

for sig_hdf5Files, l1Label in zip(sig_hdf5FilesArr, l1Labels):
    sig_dfs.append(pd.concat([pd.read_hdf(sig_hdf5, l1Label) for sig_hdf5 in sig_hdf5Files]))
    
for bkg_hdf5Files, l1Label in zip(bkg_hdf5FilesArr, l1Labels):
    bkg_dfs.append(pd.concat([pd.read_hdf(bkg_hdf5, l1Label) for bkg_hdf5 in bkg_hdf5Files]))


# plot the MET distributions
plt.hist(sig_dfs[0]['PuppiMET'], bins = 100, range = [0,200], histtype = 'step', log = True, label = "PUPPI MET")
plt.hist(sig_dfs[0]['PuppiMETNoMu'], bins = 100, range = [0,200], histtype = 'step',  label = "PUPPI MET NoMu")

for sig_df, l1Label in zip(sig_dfs, l1Labels):
    plt.hist(sig_df[l1Label], bins = 100, range = [0,200], histtype = 'step', label = l1Label)

plt.legend(fontsize=14)
plt.xlabel('L1 MET [GeV]')
plt.ylabel('Events')
plt.tight_layout()
plt.savefig("plots/MET.pdf", format="pdf")
plt.clf()


# plot the MET resolution
for sig_df, l1Label in zip(sig_dfs, l1Labels):
    plt.hist((sig_df[l1Label] - sig_df['PuppiMETNoMu']), bins = 80, range = [-100,100], label = l1Label + " Diff")

plt.legend(fontsize=14)
plt.xlabel('L1 MET - Puppi MET [GeV]')
plt.ylabel('Events')
plt.tight_layout()
plt.savefig("plots/MET_res.pdf", format="pdf")
plt.clf()

# plot the jet distributions
#plt.hist(sig_dfs[0]['Jet_pt_0'], bins = 100, range = [0,200], histtype = 'step', log = True, label = "PUPPI MET")
#plt.hist(sig_dfs[0]['Jet_pt'], bins = 100, range = [0,200], histtype = 'step',  label = "PUPPI MET NoMu")


# make fixed rate MET efficiencies

# rate plots must be in bins of GeV
ptRange = [0,200]
bins = ptRange[1]

l1METRates = []
l1METThresholdsArr = [l1METThresholds]

# get rate hist for "default" objects
rateScale = 40000000*(2452/3564)/len(bkg_dfs[0])
rateHist = plt.hist(bkg_dfs[0], bins=bins, range=ptRange, histtype = 'step', label=l1Labels[0], cumulative=-1, log=True, weights=np.full(len(bkg_dfs[0]), rateScale))

for l1METThreshold in l1METThresholds:
    # get rates for the default thresholds
    l1METRate = rateHist[0][l1METThreshold]
    l1METRates.append(l1METRate)

for i in range(1, nComp):
    # get thresholds for the fixed rates
    rateScale = 40000000*(2452/3564)/len(bkg_dfs[i])
    rateHist = plt.hist(bkg_dfs[i], bins=bins, range=ptRange, histtype = 'step', label=l1Labels[i], cumulative=-1, log=True, weights=np.full(len(bkg_dfs[i]), rateScale))
    thresholds = []
    for l1METThreshold in l1METThresholds:
        # get threshold for this rate
        thresholds.append(plotting.getThreshForRate(rateHist[0], bins, l1METRates[l1METThresholds.index(l1METThreshold)]))
    l1METThresholdsArr.append(thresholds)

plt.legend(fontsize=14)
plt.xlabel('L1 MET [GeV]')
plt.ylabel('Rate [Hz]')
plt.tight_layout()
plt.savefig("plots/MET_rates.pdf", format="pdf")
plt.clf()


# plot the MET efficiency
marks = cycle(('o', 's', '^', 'v', 'D', '*', '+', 'x'))
cols = cycle(('tab:blue','tab:orange','tab:green','tab:red','tab:purple', 'tab:pink', 'tab:cyan', 'tab:brown', 'tab:olive'))
m=0
for sig_df, l1Label, l1METThresholds in zip(sig_dfs, l1Labels, l1METThresholdsArr):
       for l1METThreshold in l1METThresholds:
              eff_data, xvals,err = plotting.efficiency(sig_df[l1Label], sig_df['PuppiMETNoMu'], l1METThreshold, 10, 400)
              plt.scatter(xvals, eff_data, label=l1Label + " > " + str(l1METThreshold), marker=next(marks), color=next(cols))
              m+=1

plt.axhline(0.95, linestyle='--', color='black')
plt.legend(fontsize=14)
plt.xlabel('PuppiMETnoMu [GeV]')
plt.ylabel('Efficiency')
plt.tight_layout()
plt.savefig("plots/MET_eff.pdf", format="pdf")
plt.clf()
