import awkward as ak
import uproot
import numpy as np
import pandas as pd

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
]

years = ["2016", "2017", "2018"]

files = [
    "root://cmseos.fnal.gov//store/user/cmantill/analyzer/v4_Jan24_ak8_option8_2016/signal/parts/GluGluToHHTo4B_node_cHHH1_TuneCUETP8M1_PSWeights_13TeV-powheg-pythia8_tree.root:Events"
]

files += [
    f"root://cmseos.fnal.gov//store/user/cmantill/analyzer/v4_Jan24_ak8_option8_{year}/signal/parts/GluGluToHHTo4B_node_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8_tree.root:Events"
    for year in years
    if year != "2016"
]

print(
    "Fraction of overlap with (resolved_veto(pt) && boosted_accept(pt) && resolved_accept)/(boosted_accept(pt))"
)

print(
    "(1): boosted_accept = fatJet1Pt>300 && fatJet2Pt>300 && fatJet1MassSD>50 && fatJet2MassRegressed>50 && ev.fatJet1PNetXbb>0.8 && ev.fatJet2PNetXbb>0.95"
)

print(
    "(2): boosted_accept = fatJet1Pt>300 && fatJet2Pt>300 && fatJet1MassSD>50 && fatJet2MassRegressed>50 && ev.fatJet1PNetXbb>0.8 && ev.fatJet2PNetXbb>0.98"
)

print(
    "(3): QCD_CR = fatJet1Pt>300 && fatJet2Pt>300 && fatJet1MassSD>50 && fatJet2MassRegressed>50 && ev.fatJet1PNetXbb>0.8 && ev.fatJet2PNetXbb<0.95"
)

events = uproot.concatenate(files, branches)

overlaps_boosted = []
overlaps_resolved = []

pt_cuts = np.arange(300, 601, 50)

for ptcut in pt_cuts:
    resolved_veto_pt = (
        (events.fatJet1Pt < ptcut)
        | (events.fatJet2Pt < ptcut)
        | (events.fatJet1MassSD_noJMS < 30)
        | (events.fatJet2MassSD_noJMS < 30)
        | (events.fatJet1NSubJets < 2)
        | (events.fatJet2NSubJets < 2)
    )

    resolved_accept = (
        (events.jet1Pt > 0) & (events.jet2Pt > 0) & (events.jet3Pt > 0) & (events.jet4Pt > 0)
    )

    # should technically include BDT selection for this?
    boosted_acceptances = []

    # boosted accept 1
    boosted_acceptances.append(
        (events.fatJet1Pt > ptcut)
        & (events.fatJet2Pt > ptcut)
        & (events.fatJet1MassSD > 50)
        & (events.fatJet2MassRegressed > 50)
        & (events.fatJet1PNetXbb > 0.8)
        & (events.fatJet2PNetXbb > 0.95)
    )

    # boosted accept 2
    boosted_acceptances.append(
        (events.fatJet1Pt > ptcut)
        & (events.fatJet2Pt > ptcut)
        & (events.fatJet1MassSD > 50)
        & (events.fatJet2MassRegressed > 50)
        & (events.fatJet1PNetXbb > 0.8)
        & (events.fatJet2PNetXbb > 0.98)
    )

    # qcd cr
    boosted_acceptances.append(
        (events.fatJet1Pt > ptcut)
        & (events.fatJet2Pt > ptcut)
        & (events.fatJet1MassSD > 50)
        & (events.fatJet2MassRegressed > 50)
        & (events.fatJet1PNetXbb > 0.8)
        & (events.fatJet2PNetXbb < 0.95)
    )

    overlaps_boosted.append(
        [
            ak.sum(resolved_veto_pt & resolved_accept & boosted_acceptances[i])
            / ak.sum(boosted_acceptances[i])
            for i in range(3)
        ]
    )

    overlaps_resolved.append(
        [
            ak.sum(resolved_veto_pt & resolved_accept & boosted_acceptances[i])
            / ak.sum(resolved_veto_pt & resolved_accept)
            for i in range(3)
        ]
    )

table = np.concatenate((pt_cuts[:, np.newaxis], overlaps_boosted), axis=1)
pddf = pd.DataFrame(table, columns=["pT cut"] + [f"Region {i + 1}" for i in range(3)])
pddf.to_csv("overlaps_boosted.csv", index=False)

table = np.concatenate((pt_cuts[:, np.newaxis], overlaps_resolved), axis=1)
pddf = pd.DataFrame(table, columns=["pT cut"] + [f"Region {i + 1}" for i in range(3)])
pddf.to_csv("overlaps_resolved.csv", index=False)
