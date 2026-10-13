# Generated data-audit statistics (`huskyquiet audit`)

Records: 193,610

## Date coverage

{
  "min_applied": "1990-03-21",
  "max_applied": "2026-10-10",
  "min_issued": "1992-06-19",
  "max_issued": "2026-10-10"
}

## applications_by_year

```
 year    n  share_never_issued
 1990    6            0.000000
 1992    8            0.000000
 1994    2            0.000000
 1995    4            0.000000
 1996    5            0.000000
 1997    5            0.000000
 1998    7            0.000000
 1999   15            0.000000
 2000   24            0.083333
 2001   38            0.026316
 2002   83            0.036145
 2003  171            0.023392
 2004  392            0.010204
 2005 3149            0.028581
 2006 8316            0.028620
 2007 7634            0.054362
 2008 6353            0.063749
 2009 4897            0.027772
 2010 5240            0.024427
 2011 5628            0.023987
 2012 5994            0.021021
 2013 7409            0.022270
 2014 7659            0.027549
 2015 8093            0.024713
 2016 8457            0.034882
 2017 8591            0.040042
 2018 8409            0.053038
 2019 7646            0.061993
 2020 6550            0.070992
 2021 7056            0.066043
 2022 6770            0.093058
 2023 5603            0.088881
 2024 6064            0.087566
 2025 6257            0.115710
 2026 5017            0.368348
```

## status_counts

```
           status_current      n
                Completed 134505
                  Expired  13765
Additional Info Requested  10256
                   Closed   8967
                   Issued   8236
                Withdrawn   5210
                 Canceled   4577
     Corrections Required   2115
                Scheduled   1895
    Application Completed    724
       Ready for Issuance    697
       Reviews In Process    610
     Awaiting Information    519
        Reviews Completed    488
         Ready for Intake    289
                   Active    268
                Initiated    223
    Inspections Completed    141
  Scheduled and Submitted     52
    Corrections Submitted     32
       Approved to Occupy     26
             Phase Issued     12
                  Pending      2
                   Denied      1
```

## by_suffix

```
permit_suffix      n  share_applied_missing  share_issued_missing  share_model_eligible
           CN 148864               0.141330              0.192175              0.858576
           EX  17603               1.000000              1.000000              0.000000
           DM  16142               0.196010              0.249969              0.803990
           BK   5182               0.058665              0.103821              0.941335
           RR   2110               0.963507              0.004739              0.036493
          NaN   1277               1.000000              1.000000              0.000000
           GR   1100               0.464545              0.553636              0.535455
           CC    792               0.116162              0.118687              0.883838
           PH    540               0.064815              0.261111              0.935185
```

## applied_missing_by_status

```
           status_current     n
                Completed 18539
Additional Info Requested 10241
                   Closed  8309
                Withdrawn  3891
                Scheduled  1895
                 Canceled   827
                  Expired   715
                   Issued   320
         Ready for Intake   287
                   Active   268
                Initiated   222
     Corrections Required   211
        Reviews Completed   197
  Scheduled and Submitted    52
       Reviews In Process    45
    Application Completed    17
       Ready for Issuance    16
     Awaiting Information     3
                  Pending     2
    Corrections Submitted     1
```

## missingness_by_eventual_issuance

```
 eventually_issued      n  est_project_cost  housing_units_added  related_mup  latitude  permit_type_desc  contractor_company_name  parent_permit_num   zoning  dwelling_unit_type  standard_plan  development_site  ready_to_issue_date  total_days_plan_review  dependent_building_true
             False   5114          0.049472             0.428823     0.860383  0.003715          0.030700                 0.896363           0.999218 0.998827            0.998827       0.998827          0.998827             0.916113                0.893430                 0.000391
              True 100055          0.074849             0.404887     0.915636  0.001239          0.014612                 0.821568           0.933257 0.773025            0.773365       0.773015          0.790405             0.561961                0.729799                 0.022568
```

## housing_category_by_eventual_issuance

```
       housing_category  issued  never_issued
  Single-Family Add/Alt   38985          1407
                    N/A   25151          2822
     Commercial Add/Alt   18991           885
         Middle Housing   14920             0
      Large Multifamily    1258             0
    Multifamily - Other     450             0
Pre-Approved DADU Plans     300             0
```

## description_admin_keywords_by_eventual_issuance

```
 eventually_issued      n   cancel  withdraw   expire  extension  embedded_date
             False   5114 0.181267   0.00919 0.003324   0.009777       0.178138
              True 100055 0.002809   0.00016 0.039000   0.002709       0.010734
```

## recent_cohort_pending_vs_issued

```
           group         permit_class    n  null_est_project_cost  null_housing_units_added  null_housing_units_removed  null_related_mup  null_permit_type_desc  null_latitude  null_zip  null_address_key  null_description  median_est_project_cost  kw_new_construction  kw_demolition  kw_adu  kw_townhouse  kw_multifamily  kw_tenant_improvement  kw_stfi  kw_change_of_use  kw_establish_use  kw_earthwork  kw_seismic  kw_solar  kw_deck_porch  kw_basement  kw_addition
          issued           Commercial 1003                    0.0                    0.4227                      0.4227            0.9860                 0.0130         0.0020    0.0548            0.0000               0.0                 100000.0               0.0189         0.0219  0.0000        0.0000          0.1017                 0.5603   0.2243            0.0967            0.0219        0.0080      0.0040    0.0010         0.0100       0.0140       0.0100
          issued           Industrial   64                    0.0                    0.2656                      0.2656            0.9375                 0.1094         0.0000    0.0156            0.0000               0.0                 273600.0               0.0781         0.1094  0.0000        0.0000          0.0156                 0.1094   0.0781            0.0156            0.0000        0.1250      0.0000    0.0000         0.0000       0.0000       0.0625
          issued        Institutional  151                    0.0                    0.2980                      0.2980            0.9669                 0.0265         0.0066    0.0662            0.0066               0.0                 112000.0               0.0464         0.0728  0.0000        0.0000          0.0066                 0.1325   0.2185            0.0199            0.0066        0.0265      0.0265    0.0000         0.0132       0.0132       0.0132
          issued          Multifamily  364                    0.0                    0.2418                      0.2418            0.8956                 0.0495         0.0000    0.0604            0.0000               0.0                 100000.0               0.0330         0.2308  0.0055        0.0495          0.7225                 0.0797   0.2390            0.0247            0.0577        0.0137      0.0275    0.0000         0.0412       0.0220       0.0192
          issued Single Family/Duplex 3545                    0.0                    0.3362                      0.3362            0.9890                 0.0302         0.0006    0.0669            0.0000               0.0                 124960.0               0.1763         0.1044  0.0976        0.0432          0.0087                 0.0003   0.3410            0.0037            0.2039        0.0144      0.0347    0.0003         0.0567       0.0559       0.1405
          issued          Vacant Land   14                    0.0                    0.1429                      0.1429            0.9286                 0.1429         0.0000    0.0000            0.0000               0.0                  65000.0               0.0000         0.2143  0.0000        0.0000          0.0714                 0.0000   0.0000            0.0000            0.0000        0.1429      0.0000    0.0000         0.0000       0.0000       0.0000
pending_or_never           Commercial  359                    0.0                    0.5070                      0.5070            0.9721                 0.0279         0.0000    0.0529            0.0000               0.0                 150000.0               0.0251         0.0390  0.0000        0.0056          0.0975                 0.4290   0.0752            0.1170            0.0529        0.0279      0.0056    0.0028         0.0028       0.0167       0.0418
pending_or_never           Industrial   52                    0.0                    0.6538                      0.6538            0.8846                 0.0962         0.0000    0.0385            0.0000               0.0                 862000.0               0.0577         0.2500  0.0000        0.0000          0.0192                 0.0577   0.1731            0.0577            0.0577        0.0769      0.0000    0.0000         0.0192       0.0000       0.0962
pending_or_never        Institutional   91                    0.0                    0.4286                      0.4286            0.8791                 0.0110         0.0000    0.0440            0.0000               0.0                 506800.0               0.0220         0.0549  0.0000        0.0000          0.0330                 0.0989   0.0330            0.0440            0.0330        0.0220      0.0110    0.0000         0.0000       0.0220       0.0659
pending_or_never          Multifamily  255                    0.0                    0.4196                      0.4196            0.8745                 0.0196         0.0039    0.0431            0.0039               0.0                 664562.0               0.2078         0.0902  0.0078        0.2588          0.5490                 0.0275   0.0392            0.0235            0.4157        0.0235      0.0118    0.0000         0.0314       0.0196       0.0588
pending_or_never Single Family/Duplex 1497                    0.0                    0.4963                      0.4963            0.9512                 0.0240         0.0000    0.0635            0.0000               0.0                 275509.0               0.2926         0.0868  0.1022        0.0294          0.0140                 0.0000   0.0628            0.0020            0.4148        0.0407      0.0047    0.0027         0.0407       0.0220       0.1336
pending_or_never          Vacant Land    6                    0.0                    1.0000                      1.0000            0.8333                 0.3333         0.0000    0.1667            0.0000               0.0                 115000.0               0.1667         0.1667  0.0000        0.0000          0.0000                 0.0000   0.0000            0.0000            0.0000        0.0000      0.0000    0.0000         0.0000       0.0000       0.1667
```

## applied_to_issued_days

```
{
  "n": 137812,
  "quantiles": {
    "0.25": 0.0,
    "0.5": 32.0,
    "0.75": 114.0,
    "0.9": 251.0,
    "0.95": 376.0,
    "0.99": 842.0
  },
  "share_same_day": 0.3163512611383624,
  "share_within_90d": 0.6977186311787072,
  "share_within_730d": 0.9862638957420253
}
```

## Null rate by field

```
days_issue_permit_city                  0.9618
parent_permit_num                       0.9500
related_mup                             0.9255
development_site                        0.8567
dwelling_unit_type                      0.8477
standard_plan                           0.8474
zoning                                  0.8474
contractor_company_name                 0.8342
total_days_plan_review                  0.8271
days_plan_review_city                   0.8271
plan_review_complete_date               0.8269
days_initial_plan_review                0.8246
initial_review_complete_date            0.8243
ready_to_issue_date                     0.7141
housing_units_added                     0.5240
housing_units_removed                   0.5240
completed_date                          0.4435
issued_date                             0.2733
expires_date                            0.2723
applied_date                            0.2379
est_project_cost                        0.1820
days_out_corrections                    0.0704
permit_type_desc                        0.0679
number_review_cycles                    0.0574
zip                                     0.0526
city                                    0.0469
state                                   0.0461
permit_suffix                           0.0066
latitude                                0.0058
longitude                               0.0058
address                                 0.0050
address_key                             0.0050
description                             0.0041
socrata_id                              0.0000
socrata_updated_at                      0.0000
socrata_version                         0.0000
permit_num                              0.0000
socrata_created_at                      0.0000
permit_class_mapped                     0.0000
permit_type_mapped                      0.0000
housing_category                        0.0000
dependent_building                      0.0000
link                                    0.0000
housing_units                           0.0000
status_current                          0.0000
permit_class                            0.0000
q_applieddate_sentinel                  0.0000
q_issueddate_sentinel                   0.0000
q_completeddate_sentinel                0.0000
q_expiresdate_sentinel                  0.0000
q_initialreviewcompletedate_sentinel    0.0000
q_planreviewcompletedate_sentinel       0.0000
q_coords_missing                        0.0000
q_readytoissuedate_sentinel             0.0000
q_issued_before_applied                 0.0000
is_model_eligible                       0.0000
```
