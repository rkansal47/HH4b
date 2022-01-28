import awkward as ak
import uproot
import numpy as np
import pandas as pd

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

events.fields


events = uproot.concatenate(files, branches)


pt_cuts = np.arange(300, 601, 50)

NUM_REGIONS = 4

# Regions 1-3 = Bins 1-3
# Region 4 = QCD CR

overlaps_boosted = []
overlaps_resolved = []

for ptcut in pt_cuts:
    fail_resolved_veto_pt = (
        (events.fatJet1Pt < ptcut)
        | (events.fatJet2Pt < ptcut)
        | (events.fatJet1MassSD_noJMS < 30)
        | (events.fatJet2MassSD_noJMS < 30)
        | (events.fatJet1NSubJets < 2)
        | (events.fatJet2NSubJets < 2)
    )

    nbtagged_jets = (
        ak.values_astype((events.jet1Pt > 0), int)
        + ak.values_astype((events.jet2Pt > 0), int)
        + ak.values_astype((events.jet3Pt > 0), int)
        + ak.values_astype((events.jet4Pt > 0), int)
    )

    resolved_accept = nbtagged_jets >= 3

    # should technically include BDT selection for this?
    boosted_acceptances = []

    # boosted category 1
    boosted_acceptances.append((events[BDT_VAR] > 0.43) & (events.fatJet2PNetXbb > 0.98))

    # boosted category 2
    boosted_acceptances.append(
        (
            ((events[BDT_VAR] < 0.43) & (events[BDT_VAR] > 0.11) & (events.fatJet2PNetXbb > 0.98))
            | ((events[BDT_VAR] > 0.43) & (events.fatJet2PNetXbb > 0.95))
        )
        & ~boosted_acceptances[0]
    )

    # boosted category 3
    boosted_acceptances.append(
        ((events[BDT_VAR] > 0.03) & (events.fatJet2PNetXbb > 0.95))
        & ~boosted_acceptances[0]
        & ~boosted_acceptances[1]
    )

    # qcd cr
    boosted_acceptances.append((events[BDT_VAR] > 0.03) & (events.fatJet2PNetXbb < 0.95))

    overlaps_boosted.append(
        [
            ak.sum(fail_resolved_veto_pt & resolved_accept & boosted_acceptances[i])
            / ak.sum(boosted_acceptances[i])
            for i in range(NUM_REGIONS)
        ]
    )

    overlaps_resolved.append(
        [
            ak.sum(fail_resolved_veto_pt & resolved_accept & boosted_acceptances[i])
            / ak.sum(fail_resolved_veto_pt & resolved_accept)
            for i in range(NUM_REGIONS)
        ]
    )


table = np.concatenate((pt_cuts[:, np.newaxis], overlaps_boosted), axis=1)
pddf = pd.DataFrame(table, columns=["pT cut"] + [f"Region {i + 1}" for i in range(NUM_REGIONS)])
pddf.to_csv("overlaps_boosted_3b.csv", index=False)

table = np.concatenate((pt_cuts[:, np.newaxis], overlaps_resolved), axis=1)
pddf = pd.DataFrame(table, columns=["pT cut"] + [f"Region {i + 1}" for i in range(NUM_REGIONS)])
pddf.to_csv("overlaps_resolved_3b.csv", index=False)
