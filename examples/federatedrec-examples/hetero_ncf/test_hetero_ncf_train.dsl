{
  "components": {
    "dataio_0": {
      "module": "DataIO",
      "input": {
        "data": {
          "data": [
            "args.train_data"
          ]
        }
      },
      "output": {
        "data": [
          "train"
        ],
        "model": [
          "dataio"
        ]
      }
    },
    "hetero_ncf_0": {
      "module": "HeteroNCF",
      "input": {
        "data": {
          "train_data": [
            "dataio_0.train"
          ]
        }
      },
      "output": {
        "data": [
          "train"
        ],
        "model": [
          "hetero_ncf"
        ]
      }
    }
  }
}