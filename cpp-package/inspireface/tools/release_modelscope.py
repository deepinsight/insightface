import os
import click
from modelscope.hub.api import HubApi


@click.command()
@click.option(
    '--model-id', 
    default='tunmxy/InspireFace',
    help='ModelScope model ID'
)
@click.option(
    '--model-dir',
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help='Local model directory path'
)
@click.option(
    '--token',
    help='ModelScope access token, if not provided will get from MODELSCOPE_TOKEN environment variable'
)
def upload_model(model_id, model_dir, token):
    """Upload model to ModelScope"""
    
    # Get token
    if not token:
        token = os.getenv('MODELSCOPE_TOKEN')
        if not token:
            click.echo("Error: No token provided and MODELSCOPE_TOKEN environment variable not set", err=True)
            raise click.Abort()
    
    try:
        # Login and upload
        api = HubApi()
        api.login(token)
        
        click.echo(f"Starting to upload model to {model_id}...")
        click.echo(f"Local directory: {model_dir}")
        
        api.push_model(
            model_id=model_id,
            model_dir=model_dir
        )
        
        click.echo("✅ Model uploaded successfully!")
        
    except Exception as e:
        click.echo(f"❌ Upload failed: {e}", err=True)
        raise click.Abort()


if __name__ == '__main__':
    upload_model()