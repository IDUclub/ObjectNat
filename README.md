# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/IDUclub/ObjectNat/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                     |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| objectnat/\_\_init\_\_.py                                |        3 |        0 |        0 |        0 |    100% |           |
| objectnat/\_api.py                                       |        4 |        0 |        0 |        0 |    100% |           |
| objectnat/\_config.py                                    |       14 |        0 |        0 |        0 |    100% |           |
| objectnat/\_version.py                                   |        1 |        0 |        0 |        0 |    100% |           |
| objectnat/methods/\_\_init\_\_.py                        |        0 |        0 |        0 |        0 |    100% |           |
| objectnat/methods/accessibility/\_\_init\_\_.py          |        3 |        0 |        0 |        0 |    100% |           |
| objectnat/methods/accessibility/\_utils.py               |      139 |       28 |       64 |       25 |     72% |33, 69, 82-\>86, 84, 91, 96-103, 125, 133, 137, 159, 162, 164, 166, 168, 170, 174, 179-\>182, 189, 195, 201, 227, 241, 246, 266, 269 |
| objectnat/methods/accessibility/coverage.py              |       69 |       17 |       34 |       16 |     68% |111, 113, 115, 117, 143, 150, 153-154, 161-162, 176, 292, 294, 296, 298, 300, 320, 322-\>325 |
| objectnat/methods/accessibility/isochrones.py            |       68 |       13 |       32 |       13 |     74% |104, 106, 108, 110, 142, 164, 174, 278, 280, 282, 284, 286, 306 |
| objectnat/methods/accessibility/radius.py                |       14 |        0 |        0 |        0 |    100% |           |
| objectnat/methods/noise/\_\_init\_\_.py                  |        3 |        0 |        0 |        0 |    100% |           |
| objectnat/methods/noise/noise\_init\_data.py             |        3 |        0 |        0 |        0 |    100% |           |
| objectnat/methods/noise/noise\_reduce.py                 |       72 |        1 |       26 |        2 |     97% |63, 104-\>exit |
| objectnat/methods/noise/noise\_simulation.py             |      364 |       32 |      138 |       27 |     88% |117, 121, 127, 132, 142, 146, 161-164, 172, 177, 256, 301, 308, 312, 320, 322, 327, 330, 336, 345, 350, 361, 408, 420-\>450, 426-427, 435-\>429, 438, 495, 509-510, 527 |
| objectnat/methods/noise/noise\_simulation\_simplified.py |      107 |        9 |       42 |        7 |     89% |107, 109, 111, 122-124, 201, 206, 241 |
| objectnat/methods/provision/\_\_init\_\_.py              |        2 |        0 |        0 |        0 |    100% |           |
| objectnat/methods/provision/provision.py                 |       19 |        0 |        0 |        0 |    100% |           |
| objectnat/methods/provision/provision\_calculating.py    |      195 |        8 |       58 |        8 |     94% |25, 27, 41, 43, 102, 107, 129, 136 |
| objectnat/methods/provision/provision\_exceptions.py     |       36 |        0 |       16 |        0 |    100% |           |
| objectnat/methods/provision/provision\_result.py         |       73 |        8 |       38 |        8 |     86% |52, 54, 56, 108, 130, 134, 136, 168 |
| objectnat/methods/utils/\_\_init\_\_.py                  |        0 |        0 |        0 |        0 |    100% |           |
| objectnat/methods/utils/geom\_utils.py                   |       90 |        1 |       18 |        1 |     98% |        72 |
| objectnat/methods/visibility/\_\_init\_\_.py             |        1 |        0 |        0 |        0 |    100% |           |
| objectnat/methods/visibility/visibility\_analysis.py     |      148 |       19 |       48 |       14 |     83% |50-52, 101, 139-143, 165, 174-\>178, 175-\>174, 225, 235, 243-\>246, 287, 293, 388, 391, 402, 428-429 |
| **TOTAL**                                                | **1428** |  **136** |  **514** |  **121** | **87%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/IDUclub/ObjectNat/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/IDUclub/ObjectNat/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/IDUclub/ObjectNat/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/IDUclub/ObjectNat/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2FIDUclub%2FObjectNat%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/IDUclub/ObjectNat/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.