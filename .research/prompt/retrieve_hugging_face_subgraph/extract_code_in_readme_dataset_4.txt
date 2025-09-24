
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
license: mit
multilinguality:
  - multilingual
source_datasets:
  - original
task_categories:
  - text-classification
  - token-classification
  - question-answering
  - summarization
  - text-generation
task_ids:
  - sentiment-analysis
  - topic-classification
  - named-entity-recognition
  - language-modeling
  - text-scoring
  - multi-class-classification
  - multi-label-classification
  - extractive-qa
  - news-articles-summarization
---


# Bittensor Subnet 13 Reddit Dataset

<center>
    <img src="https://huggingface.co/datasets/macrocosm-os/images/resolve/main/bittensor.png" alt="Data-universe: The finest collection of social media data the web has to offer">
</center>

<center>
    <img src="https://huggingface.co/datasets/macrocosm-os/images/resolve/main/macrocosmos-black.png" alt="Data-universe: The finest collection of social media data the web has to offer">
</center>


## Dataset Description

- **Repository:** coldmind/reddit_dataset_94
- **Subnet:** Bittensor Subnet 13
- **Miner Hotkey:** 5CCrb9H6LoDjDFKNfKqiBuLe9NNUcCqkruQ2fpUY4R1RzCkb

### Dataset Summary

This dataset is part of the Bittensor Subnet 13 decentralized network, containing preprocessed Reddit data. The data is continuously updated by network miners, providing a real-time stream of Reddit content for various analytical and machine learning tasks.
For more information about the dataset, please visit the [official repository](https://github.com/macrocosm-os/data-universe).

### Supported Tasks

The versatility of this dataset allows researchers and data scientists to explore various aspects of social media dynamics and develop innovative applications. Users are encouraged to leverage this data creatively for their specific research or business needs.
For example:

- Sentiment Analysis
- Topic Modeling
- Community Analysis
- Content Categorization

### Languages

Primary language: Datasets are mostly English, but can be multilingual due to decentralized ways of creation.

## Dataset Structure

### Data Instances

Each instance represents a single Reddit post or comment with the following fields:


### Data Fields

- `text` (string): The main content of the Reddit post or comment.
- `label` (string): Sentiment or topic category of the content.
- `dataType` (string): Indicates whether the entry is a post or a comment.
- `communityName` (string): The name of the subreddit where the content was posted.
- `datetime` (string): The date when the content was posted or commented.
- `username_encoded` (string): An encoded version of the username to maintain user privacy.
- `url_encoded` (string): An encoded version of any URLs included in the content.

### Data Splits

This dataset is continuously updated and does not have fixed splits. Users should create their own splits based on their requirements and the data's timestamp.

## Dataset Creation

### Source Data

Data is collected from public posts and comments on Reddit, adhering to the platform's terms of service and API usage guidelines.

### Personal and Sensitive Information

All usernames and URLs are encoded to protect user privacy. The dataset does not intentionally include personal or sensitive information.

## Considerations for Using the Data

### Social Impact and Biases

Users should be aware of potential biases inherent in Reddit data, including demographic and content biases. This dataset reflects the content and opinions expressed on Reddit and should not be considered a representative sample of the general population.

### Limitations

- Data quality may vary due to the nature of media sources.
- The dataset may contain noise, spam, or irrelevant content typical of social media platforms.
- Temporal biases may exist due to real-time collection methods.
- The dataset is limited to public subreddits and does not include private or restricted communities.

## Additional Information

### Licensing Information

The dataset is released under the MIT license. The use of this dataset is also subject to Reddit Terms of Use.

### Citation Information

If you use this dataset in your research, please cite it as follows:

```
@misc{coldmind2025datauniversereddit_dataset_94,
        title={The Data Universe Datasets: The finest collection of social media data the web has to offer},
        author={coldmind},
        year={2025},
        url={https://huggingface.co/datasets/coldmind/reddit_dataset_94},
        }
```

### Contributions

To report issues or contribute to the dataset, please contact the miner or use the Bittensor Subnet 13 governance mechanisms.

## Dataset Statistics

[This section is automatically updated]

- **Total Instances:** 21824370
- **Date Range:** 2013-12-04T00:00:00Z to 2025-05-02T00:00:00Z
- **Last Updated:** 2025-07-30T23:05:14Z

### Data Distribution

- Posts: 9.06%
- Comments: 90.94%

### Top 10 Subreddits

For full statistics, please refer to the `stats.json` file in the repository.

| Rank | Topic | Total Count | Percentage |
|------|-------|-------------|-------------|
| 1 | r/politics | 824494 | 3.78% |
| 2 | r/wallstreetbets | 643103 | 2.95% |
| 3 | r/worldnews | 556466 | 2.55% |
| 4 | r/CryptoCurrency | 238617 | 1.09% |
| 5 | r/Bitcoin | 231563 | 1.06% |
| 6 | r/CryptoMarkets | 85877 | 0.39% |
| 7 | r/StockMarket | 79627 | 0.36% |
| 8 | r/news | 67152 | 0.31% |
| 9 | r/trump | 57323 | 0.26% |
| 10 | r/solana | 48967 | 0.22% |


## Update History

| Date | New Instances | Total Instances |
|------|---------------|-----------------|
| 2025-02-15T09:49:18Z | 8474848 | 8474848 |
| 2025-02-18T22:16:18Z | 1078051 | 9552899 |
| 2025-02-22T10:39:38Z | 1037160 | 10590059 |
| 2025-02-24T14:05:32Z | 536637 | 11126696 |
| 2025-02-25T08:06:07Z | 135821 | 11262517 |
| 2025-02-26T02:06:38Z | 117694 | 11380211 |
| 2025-02-26T20:15:38Z | 98188 | 11478399 |
| 2025-02-27T13:54:13Z | 115218 | 11593617 |
| 2025-02-28T07:54:42Z | 106284 | 11699901 |
| 2025-03-01T01:10:19Z | 102904 | 11802805 |
| 2025-03-01T19:10:53Z | 96436 | 11899241 |
| 2025-03-02T13:18:18Z | 119708 | 12018949 |
| 2025-03-03T07:31:13Z | 133574 | 12152523 |
| 2025-03-04T01:32:06Z | 109735 | 12262258 |
| 2025-03-04T18:43:38Z | 97350 | 12359608 |
| 2025-03-05T12:44:15Z | 130919 | 12490527 |
| 2025-03-06T06:45:14Z | 126925 | 12617452 |
| 2025-03-07T00:43:30Z | 104732 | 12722184 |
| 2025-03-07T18:44:07Z | 116505 | 12838689 |
| 2025-03-08T12:47:47Z | 117328 | 12956017 |
| 2025-03-09T06:48:20Z | 109517 | 13065534 |
| 2025-03-10T00:48:53Z | 93437 | 13158971 |
| 2025-03-10T18:45:10Z | 101512 | 13260483 |
| 2025-03-11T12:45:47Z | 108362 | 13368845 |
| 2025-03-12T06:44:09Z | 121285 | 13490130 |
| 2025-03-13T00:58:07Z | 102076 | 13592206 |
| 2025-03-13T19:03:30Z | 99404 | 13691610 |
| 2025-03-14T12:25:59Z | 115115 | 13806725 |
| 2025-03-15T06:41:25Z | 133833 | 13940558 |
| 2025-03-16T01:03:10Z | 100075 | 14040633 |
| 2025-03-16T19:31:50Z | 99287 | 14139920 |
| 2025-03-17T13:49:53Z | 116570 | 14256490 |
| 2025-03-18T08:10:30Z | 138306 | 14394796 |
| 2025-03-19T01:52:10Z | 86577 | 14481373 |
| 2025-03-19T19:55:33Z | 133642 | 14615015 |
| 2025-03-20T14:14:42Z | 141070 | 14756085 |
| 2025-03-21T08:16:12Z | 164718 | 14920803 |
| 2025-03-22T02:20:10Z | 149922 | 15070725 |
| 2025-03-22T19:39:17Z | 104121 | 15174846 |
| 2025-03-23T13:53:39Z | 130923 | 15305769 |
| 2025-03-24T08:08:17Z | 145347 | 15451116 |
| 2025-03-25T02:23:59Z | 144164 | 15595280 |
| 2025-03-25T20:06:51Z | 120769 | 15716049 |
| 2025-03-26T14:04:50Z | 146447 | 15862496 |
| 2025-03-27T07:20:11Z | 139559 | 16002055 |
| 2025-03-28T01:30:35Z | 1 | 16002056 |
| 2025-03-28T18:40:29Z | 97564 | 16099620 |
| 2025-03-29T12:51:19Z | 117575 | 16217195 |
| 2025-03-30T07:04:22Z | 116324 | 16333519 |
| 2025-03-31T01:34:17Z | 112621 | 16446140 |
| 2025-03-31T19:45:09Z | 107176 | 16553316 |
| 2025-04-01T13:13:39Z | 116745 | 16670061 |
| 2025-04-02T07:29:35Z | 143290 | 16813351 |
| 2025-04-03T01:46:49Z | 139770 | 16953121 |
| 2025-04-03T20:04:38Z | 113948 | 17067069 |
| 2025-04-04T13:21:34Z | 115606 | 17182675 |
| 2025-04-05T07:41:24Z | 137915 | 17320590 |
| 2025-04-06T02:01:14Z | 109575 | 17430165 |
| 2025-04-06T20:12:36Z | 110043 | 17540208 |
| 2025-04-07T14:34:07Z | 135247 | 17675455 |
| 2025-04-08T08:29:31Z | 134719 | 17810174 |
| 2025-04-09T02:52:00Z | 145247 | 17955421 |
| 2025-04-09T21:14:06Z | 128507 | 18083928 |
| 2025-04-10T15:37:02Z | 128049 | 18211977 |
| 2025-04-11T10:21:43Z | 144272 | 18356249 |
| 2025-04-12T04:44:31Z | 170449 | 18526698 |
| 2025-04-12T22:52:11Z | 126456 | 18653154 |
| 2025-04-13T16:30:20Z | 105975 | 18759129 |
| 2025-04-14T10:50:01Z | 149919 | 18909048 |
| 2025-04-15T05:11:48Z | 146760 | 19055808 |
| 2025-04-15T22:32:31Z | 122441 | 19178249 |
| 2025-04-16T16:57:22Z | 132324 | 19310573 |
| 2025-04-17T02:17:06Z | 60715 | 19371288 |
| 2025-04-17T19:44:32Z | 110784 | 19482072 |
| 2025-04-18T14:00:57Z | 126984 | 19609056 |
| 2025-04-19T08:18:14Z | 136725 | 19745781 |
| 2025-04-20T02:35:34Z | 123778 | 19869559 |
| 2025-04-20T20:52:44Z | 108298 | 19977857 |
| 2025-04-21T15:13:10Z | 116778 | 20094635 |
| 2025-04-22T09:36:35Z | 147418 | 20242053 |
| 2025-04-23T04:02:37Z | 127237 | 20369290 |
| 2025-04-23T22:30:32Z | 122823 | 20492113 |
| 2025-04-24T16:59:23Z | 108939 | 20601052 |
| 2025-04-25T11:28:39Z | 138150 | 20739202 |
| 2025-04-26T05:56:51Z | 159930 | 20899132 |
| 2025-04-27T00:23:25Z | 117714 | 21016846 |
| 2025-04-27T18:51:23Z | 108703 | 21125549 |
| 2025-04-28T13:19:44Z | 124254 | 21249803 |
| 2025-04-29T07:47:28Z | 140740 | 21390543 |
| 2025-04-30T02:16:29Z | 108675 | 21499218 |
| 2025-04-30T20:45:47Z | 103911 | 21603129 |
| 2025-05-01T15:15:13Z | 147068 | 21750197 |
| 2025-05-02T09:22:53Z | 74057 | 21824254 |
| 2025-05-03T03:52:34Z | 1 | 21824255 |
| 2025-05-03T22:16:57Z | 1 | 21824256 |
| 2025-05-04T16:39:33Z | 1 | 21824257 |
| 2025-05-05T10:58:53Z | 1 | 21824258 |
| 2025-05-06T21:06:52Z | 1 | 21824259 |
| 2025-05-07T15:37:58Z | 1 | 21824260 |
| 2025-05-08T10:07:25Z | 1 | 21824261 |
| 2025-05-09T04:31:24Z | 1 | 21824262 |
| 2025-05-09T22:56:11Z | 1 | 21824263 |
| 2025-05-10T17:24:51Z | 1 | 21824264 |
| 2025-05-11T11:55:08Z | 1 | 21824265 |
| 2025-05-12T06:22:44Z | 1 | 21824266 |
| 2025-05-13T00:50:58Z | 1 | 21824267 |
| 2025-05-13T19:22:09Z | 1 | 21824268 |
| 2025-05-14T13:50:28Z | 1 | 21824269 |
| 2025-05-15T08:14:17Z | 1 | 21824270 |
| 2025-05-16T02:36:37Z | 1 | 21824271 |
| 2025-05-16T21:04:27Z | 1 | 21824272 |
| 2025-05-17T15:34:22Z | 1 | 21824273 |
| 2025-05-18T10:05:04Z | 1 | 21824274 |
| 2025-05-19T04:35:45Z | 1 | 21824275 |
| 2025-05-19T23:05:15Z | 1 | 21824276 |
| 2025-05-20T17:30:24Z | 1 | 21824277 |
| 2025-05-21T11:55:19Z | 1 | 21824278 |
| 2025-05-22T06:19:32Z | 1 | 21824279 |
| 2025-05-23T00:42:20Z | 1 | 21824280 |
| 2025-05-23T19:05:46Z | 1 | 21824281 |
| 2025-05-24T13:27:38Z | 1 | 21824282 |
| 2025-05-25T07:50:49Z | 1 | 21824283 |
| 2025-05-26T02:13:44Z | 1 | 21824284 |
| 2025-05-26T20:34:58Z | 1 | 21824285 |
| 2025-05-27T14:57:11Z | 1 | 21824286 |
| 2025-05-28T09:20:49Z | 1 | 21824287 |
| 2025-05-29T03:45:57Z | 1 | 21824288 |
| 2025-05-29T22:16:02Z | 1 | 21824289 |
| 2025-05-30T16:46:25Z | 1 | 21824290 |
| 2025-05-31T11:17:20Z | 1 | 21824291 |
| 2025-06-01T05:45:48Z | 1 | 21824292 |
| 2025-06-02T00:14:25Z | 1 | 21824293 |
| 2025-06-02T18:44:41Z | 1 | 21824294 |
| 2025-06-03T13:16:52Z | 1 | 21824295 |
| 2025-06-04T07:48:43Z | 1 | 21824296 |
| 2025-06-05T02:21:17Z | 1 | 21824297 |
| 2025-06-05T20:51:57Z | 1 | 21824298 |
| 2025-06-06T15:20:33Z | 1 | 21824299 |
| 2025-06-07T09:48:30Z | 1 | 21824300 |
| 2025-06-08T04:15:11Z | 1 | 21824301 |
| 2025-06-08T22:41:06Z | 1 | 21824302 |
| 2025-06-09T17:05:49Z | 1 | 21824303 |
| 2025-06-10T11:31:55Z | 1 | 21824304 |
| 2025-06-11T05:55:05Z | 1 | 21824305 |
| 2025-06-12T00:17:10Z | 1 | 21824306 |
| 2025-06-12T18:41:17Z | 1 | 21824307 |
| 2025-06-13T13:04:33Z | 1 | 21824308 |
| 2025-06-14T07:27:06Z | 1 | 21824309 |
| 2025-06-15T01:51:09Z | 1 | 21824310 |
| 2025-06-15T20:13:24Z | 1 | 21824311 |
| 2025-06-16T14:33:20Z | 1 | 21824312 |
| 2025-06-17T08:56:32Z | 1 | 21824313 |
| 2025-06-18T03:19:17Z | 1 | 21824314 |
| 2025-06-18T21:45:55Z | 1 | 21824315 |
| 2025-06-19T16:15:49Z | 1 | 21824316 |
| 2025-06-20T10:46:31Z | 1 | 21824317 |
| 2025-06-21T05:17:24Z | 1 | 21824318 |
| 2025-06-21T23:46:54Z | 1 | 21824319 |
| 2025-06-22T18:17:23Z | 1 | 21824320 |
| 2025-06-23T12:48:43Z | 1 | 21824321 |
| 2025-06-24T07:18:12Z | 1 | 21824322 |
| 2025-06-25T01:48:06Z | 1 | 21824323 |
| 2025-06-25T20:19:49Z | 1 | 21824324 |
| 2025-06-26T14:50:18Z | 1 | 21824325 |
| 2025-06-27T09:19:04Z | 1 | 21824326 |
| 2025-06-28T03:48:57Z | 1 | 21824327 |
| 2025-06-28T22:18:27Z | 1 | 21824328 |
| 2025-06-29T16:47:55Z | 1 | 21824329 |
| 2025-06-30T11:19:43Z | 1 | 21824330 |
| 2025-07-01T05:47:27Z | 1 | 21824331 |
| 2025-07-02T00:12:09Z | 1 | 21824332 |
| 2025-07-02T18:39:50Z | 1 | 21824333 |
| 2025-07-03T13:10:34Z | 1 | 21824334 |
| 2025-07-04T07:41:56Z | 1 | 21824335 |
| 2025-07-05T02:09:05Z | 1 | 21824336 |
| 2025-07-05T20:31:30Z | 1 | 21824337 |
| 2025-07-06T14:53:05Z | 1 | 21824338 |
| 2025-07-07T09:14:30Z | 1 | 21824339 |
| 2025-07-08T03:34:59Z | 1 | 21824340 |
| 2025-07-08T21:55:46Z | 1 | 21824341 |
| 2025-07-09T16:15:10Z | 1 | 21824342 |
| 2025-07-10T10:33:21Z | 1 | 21824343 |
| 2025-07-11T04:52:37Z | 1 | 21824344 |
| 2025-07-11T23:10:16Z | 1 | 21824345 |
| 2025-07-12T17:28:25Z | 1 | 21824346 |
| 2025-07-13T11:47:20Z | 1 | 21824347 |
| 2025-07-14T06:05:12Z | 1 | 21824348 |
| 2025-07-15T00:22:56Z | 1 | 21824349 |
| 2025-07-15T18:38:19Z | 1 | 21824350 |
| 2025-07-16T12:51:15Z | 1 | 21824351 |
| 2025-07-17T07:03:55Z | 1 | 21824352 |
| 2025-07-18T01:16:42Z | 1 | 21824353 |
| 2025-07-18T19:29:40Z | 1 | 21824354 |
| 2025-07-19T13:42:18Z | 1 | 21824355 |
| 2025-07-20T07:55:09Z | 1 | 21824356 |
| 2025-07-21T02:07:58Z | 1 | 21824357 |
| 2025-07-21T20:20:52Z | 1 | 21824358 |
| 2025-07-22T14:34:02Z | 1 | 21824359 |
| 2025-07-23T08:47:00Z | 1 | 21824360 |
| 2025-07-24T03:00:03Z | 1 | 21824361 |
| 2025-07-24T21:15:32Z | 1 | 21824362 |
| 2025-07-25T15:29:59Z | 1 | 21824363 |
| 2025-07-26T09:42:55Z | 1 | 21824364 |
| 2025-07-27T03:57:06Z | 1 | 21824365 |
| 2025-07-27T22:10:53Z | 1 | 21824366 |
| 2025-07-28T16:23:44Z | 1 | 21824367 |
| 2025-07-29T10:37:41Z | 1 | 21824368 |
| 2025-07-30T04:51:10Z | 1 | 21824369 |
| 2025-07-30T23:05:14Z | 1 | 21824370 |

Output:
{
    "extracted_code": ""
}
