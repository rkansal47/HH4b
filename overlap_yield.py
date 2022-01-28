import awkward as ak
import uproot
import numpy as np
from coffea.hist import clopper_pearson_interval

# import pandas as pd


BDT_VAR = "disc_qcd_and_ttbar_Run2_enhanced_v8p2"

branches = [
    "fatJet1Pt",
    "fatJet1MassSD",
    "fatJet1MassRegressed",
    "fatJet1MassSD_noJMS",  # this is the softdrop mass straight out of nanoAOD w/o any JMS corrections
    "fatJet1PNetXbb",
    "fatJet2Pt",
    "fatJet2MassSD",
    "fatJet2MassRegressed",
    "fatJet2MassSD_noJMS",
    "fatJet2PNetXbb",
    "jet1Pt",
    "jet2Pt",
    "jet3Pt",
    "jet4Pt",  # forgot to save nbtaggedejets but all these jets* have deepFlavB medium working point
    "xsecWeight",
    "fatJet1NSubJets",
    "fatJet2NSubJets",
    BDT_VAR,
]

years = ["2016", "2017", "2018"]

files = [
    "root://cmseos.fnal.gov//store/group/lpcdihiggsboost/sixie/analyzer/HHTo4BNtupler/20220124/option5/combined/2016/GluGluToHHTo4B_node_cHHH1_TuneCUETP8M1_PSWeights_13TeV-powheg-pythia8_tree_BDTs.root",
    "root://cmseos.fnal.gov//store/group/lpcdihiggsboost/sixie/analyzer/HHTo4BNtupler/20220124/option5/combined/2017/GluGluToHHTo4B_node_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8_tree_BDTs.root",
    "root://cmseos.fnal.gov//store/group/lpcdihiggsboost/sixie/analyzer/HHTo4BNtupler/20220124/option5/combined/2018/GluGluToHHTo4B_node_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8_tree_BDTs.root",
]

events = uproot.concatenate(files, branches)
# events_full = uproot.concatenate(files)

nbtagged_jets = (
    ak.values_astype((events.jet1Pt > 0), int)
    + ak.values_astype((events.jet2Pt > 0), int)
    + ak.values_astype((events.jet3Pt > 0), int)
    + ak.values_astype((events.jet4Pt > 0), int)
)

resolved_accept = nbtagged_jets >= 3

pt_cuts = np.arange(300, 601, 50)

NUM_REGIONS = 4

# Regions 1-3 = Bins 1-3
# Region 4 = QCD CR

numerators = []
denominators_boosted = []
denominators_resolved = []

tag = "3b_bdt"

for ptcut in pt_cuts:
    fail_resolved_veto_pt = (
        (events.fatJet1Pt < ptcut)
        | (events.fatJet2Pt < ptcut)
        | (events.fatJet1MassSD_noJMS < 30)
        | (events.fatJet2MassSD_noJMS < 30)
        | (events.fatJet1NSubJets < 2)
        | (events.fatJet2NSubJets < 2)
    )

    # should technically include BDT selection for this?
    boosted_acceptances = []

    baseline_cut = (
        (events.fatJet1Pt > ptcut)
        & (events.fatJet2Pt > ptcut)
        & (events.fatJet1MassSD > 50)
        & (events.fatJet2MassRegressed > 50)
        & (events.fatJet1PNetXbb > 0.8)
    )

    # boosted category 1
    boosted_acceptances.append(
        (events[BDT_VAR] > 0.43) & (events.fatJet2PNetXbb > 0.98) & baseline_cut
    )

    # boosted category 2
    boosted_acceptances.append(
        (
            ((events[BDT_VAR] < 0.43) & (events[BDT_VAR] > 0.11) & (events.fatJet2PNetXbb > 0.98))
            | ((events[BDT_VAR] > 0.43) & (events.fatJet2PNetXbb > 0.95))
        )
        & ~boosted_acceptances[0]
        & baseline_cut
    )

    # boosted category 3
    boosted_acceptances.append(
        ((events[BDT_VAR] > 0.03) & (events.fatJet2PNetXbb > 0.95))
        & ~boosted_acceptances[0]
        & ~boosted_acceptances[1]
        & baseline_cut
    )

    # qcd cr
    boosted_acceptances.append(
        (events[BDT_VAR] > 0.03) & (events.fatJet2PNetXbb < 0.95) & baseline_cut
    )

    numerators.append(
        [
            ak.sum(fail_resolved_veto_pt & resolved_accept & boosted_acceptances[i])
            for i in range(NUM_REGIONS)
        ]
    )

    denominators_boosted.append([ak.sum(boosted_acceptances[i]) for i in range(NUM_REGIONS)])
    denominators_resolved.append(
        [ak.sum(fail_resolved_veto_pt & resolved_accept) for i in range(NUM_REGIONS)]
    )


numerators = np.array(numerators)
denominators_boosted = np.array(denominators_boosted)
denominators_resolved = np.array(denominators_resolved)

overlaps_boosted = numerators / denominators_boosted
overlaps_resolved = numerators / denominators_resolved

cp_boosted = clopper_pearson_interval(numerators, denominators_boosted)
errors_boosted = (cp_boosted[1] - cp_boosted[0]) / 2
cp_resolved = clopper_pearson_interval(numerators, denominators_resolved)
errors_resolved = (cp_resolved[1] - cp_resolved[0]) / 2

######################################
# Table Formatting
######################################

decimals_boosted = -(np.floor(np.log10(errors_boosted))).astype(int)
decimals_boosted[errors_boosted == 0] = 0
decimals_boosted -= ((errors_boosted * 10 ** decimals_boosted) >= 9.5).astype(int)
decimals_boosted = np.clip(decimals_boosted, a_min=0, a_max=3)

output_list_boosted = []
for i in range(overlaps_boosted.shape[0]):
    row = []
    for j in range(overlaps_boosted.shape[1]):
        row.append(
            f"{overlaps_boosted[i][j]:.{decimals_boosted[i][j]}f}±{errors_boosted[i][j]:.{decimals_boosted[i][j]}f}"
            if errors_boosted[i][j] >= 0.0005
            else "0"
        )
    output_list_boosted.append(row)

np.savetxt(f"overlaps_boosted_{tag}.csv", np.array(output_list_boosted), fmt="%s", delimiter=",")

decimals_resolved = -(np.floor(np.log10(errors_resolved))).astype(int)
decimals_resolved[errors_resolved == 0] = 0
decimals_resolved -= ((errors_resolved * 10 ** decimals_resolved) >= 9.5).astype(int)
decimals_resolved = np.clip(decimals_resolved, a_min=0, a_max=3)

output_list_resolved = []
for i in range(overlaps_resolved.shape[0]):
    row = []
    for j in range(overlaps_resolved.shape[1]):
        row.append(
            f"{overlaps_resolved[i][j]:.{decimals_resolved[i][j]}f}±{errors_resolved[i][j]:.{decimals_resolved[i][j]}f}"
            if errors_resolved[i][j] >= 0.0005
            else "0"
        )
    output_list_resolved.append(row)

np.savetxt(f"overlaps_resolved_{tag}.csv", np.array(output_list_resolved), fmt="%s", delimiter=",")

# table = np.concatenate((pt_cuts[:, np.newaxis], overlaps_boosted), axis=1)
# pddf = pd.DataFrame(table, columns=["pT cut"] + [f"Region {i + 1}" for i in range(NUM_REGIONS)])
# pddf.to_csv(f"overlaps_boosted_{tag}.csv", index=False)
#
# table = np.concatenate((pt_cuts[:, np.newaxis], overlaps_resolved), axis=1)
# pddf = pd.DataFrame(table, columns=["pT cut"] + [f"Region {i + 1}" for i in range(NUM_REGIONS)])
# pddf.to_csv(f"overlaps_resolved_{tag}.csv", index=False)
