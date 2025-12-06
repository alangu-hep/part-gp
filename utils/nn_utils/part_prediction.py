import functools
import glob
from weaver.utils.logger import _logger, warn_n_times
from weaver.utils.dataset import SimpleIterDataset
from torch.utils.data import DataLoader

# Adapted from Weaver w/ demo feature added
def test_load(args):
    """
    Loads the test data.
    :param args:
    :return: test_loaders, data_config
    """
    # keyword-based --data-test: 'a:/path/to/a b:/path/to/b'
    # split --data-test: 'a%10:/path/to/a/*'
    file_dict = {}
    split_dict = {}
    for f in args.data_test:
        if ':' in f:
            name, fp = f.split(':')
            if '%' in name:
                name, split = name.split('%')
                split_dict[name] = int(split)
        else:
            name, fp = '', f
        files = glob.glob(fp)
        if name in file_dict:
            file_dict[name] += files
        else:
            file_dict[name] = files

    # sort files
    for name, files in file_dict.items():
        file_dict[name] = sorted(files)

    # apply splitting
    for name, split in split_dict.items():
        files = file_dict.pop(name)
        for i in range((len(files) + split - 1) // split):
            file_dict[f'{name}_{i}'] = files[i * split:(i + 1) * split]
    
    def get_test_loader(name):
        filelist = file_dict[name]

        if args.demo:
            import fnmatch
            prereq = '*_000.root'
            filelist = fnmatch.filter(filelist, prereq)
        
        _logger.info('Running on test file group %s with %d files:\n...%s', name, len(filelist), '\n...'.join(filelist))
        num_workers = min(args.num_workers, len(filelist))
        test_data = SimpleIterDataset({name: filelist}, args.data_config, for_training=False,
                                      extra_selection=args.extra_test_selection,
                                      load_range_and_fraction=((0, 1), args.data_fraction),
                                      fetch_by_files=True, fetch_step=1,
                                      name='test_' + name)
        test_loader = DataLoader(test_data, num_workers=num_workers, batch_size=args.batch_size, drop_last=False,
                                 pin_memory=True)
        return test_loader

    test_loaders = {name: functools.partial(get_test_loader, name) for name in file_dict}
    data_config = SimpleIterDataset({}, args.data_config, for_training=False).config
    return test_loaders, data_config