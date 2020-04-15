from pathlib import Path
import argparse


from data_loading import dataset


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-which_dataset', dest="which_dataset",
                        help='mmwhs/imatfib/acdc', default='mmwhs')
    parser.add_argument('-seg_type', dest="seg_type",
                        help='multiple-classes/whole-heart', default='multiple-classes')

    args = parser.parse_args()
    print(args)

    images_path, gts_path = create_paths(args.which_dataset, args.seg_type)
    dset = dataset.MRI_Dataset(images_path, gts_path)

    for elem in dset:
        print(elem)


def create_paths(which_dataset, seg_type):
    """
    Args:

    """
    if which_dataset == 'mmwhs':
        mmwhs_path = Path.cwd() / 'datasets' / 'mmwhs'
        images_path = mmwhs_path / 'image'
        gts_path = mmwhs_path / 'ground-truth' / seg_type

    return images_path, gts_path


if __name__ == '__main__':
    main()
