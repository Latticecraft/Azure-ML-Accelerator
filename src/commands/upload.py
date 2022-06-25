import os, argparse
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__


def main(args):
    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    print(f'connection string: {connect_str}')

    filename = args.file_path.split('/')[-1]

    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    blob_client = blob_service_client.get_blob_client(container='input', blob=f'{args.project}/{filename}')

    with open(args.file_path, "rb") as data:
        blob_client.upload_blob(data)


def prompt_input(args, name, prompt):
    if getattr(args, name) is None:
        setattr(args, name, input(prompt))
        if getattr(args, name) is None:
            exit(1)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--file-path', type=str, required=False)
    parser.add_argument('--project', type=str, required=False)
    
    # parse args
    args = parser.parse_args()

    prompt_input(args, 'file_path', 'File to upload: ')
    prompt_input(args, 'project', 'Project: ')
        
    # return args
    return args


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)
