## Data Preparation

You should organize your data according to the following structure.

```
data/
├── sft/
│   ├── images.zip
│   └── json
│       ├── sft_part_0.json
│       ├── sft_part_1.json
│       ├── sft_part_2.json
│       ├── sft_part_3.json
│       └── sft_part_4.json
└── rl/
│   ├── perception_all_1.parquet
│   ├── perception_all_2.parquet
│   ├── perception_all_3.parquet
│   ├── perception_all_4.parquet
│   ├── perception_all_5.parquet
│   ├── reason.parquet
│   └── search.parquet
└── search_cache/
    ├── fvqa_train_image_search_results_cache.json
    └── cached_images
        └── train.zip

```


### Cold Start Data
You should firstly unzip the `images.zip` by using the following command.

```bash
cd sft
unzip images.zip
```

After that, you can run `data_convert.py` to convert the json data.

```bash
python ../cold_start/data_convert.py --input_path path_to_json_path --data_path path_to_image_path
```

It is worth noting that we do not provide the multimodal CoT SFT data due to policy reasons.

### Search Cache

You should firstly unzip the `train.zip`.

```bash
cd search_cache/cached_images
unzip train.zip
```

Then, you should run `cache_convert.py` to convert the json data.


```bash
python ../reinforcement_learning/cache_convert.py --input_json_path path_to_json_path --data_path path_to_image_path
```


