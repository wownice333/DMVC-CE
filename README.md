## Mixture of Experts as Representation Learner for Deep Multi-view Clustering
This is the official implementation of Mixture of Experts as Representation Learner for Deep Multi-view Clustering.

### Dependencies

- python 3.8, pytorch, numpy, scikit-learn, pandas, tqdm

If you have installed above mentioned packages you can skip this step. Otherwise run:

    pip install -r requirements.txt

## Reproduce multi-view results

To generate results

    python demo_DMVC_CE.py --DS HW --eval True

To train LG-FGAD without loading saved weight files

    python demo_DMVC_CE.py --DS HW --eval False

