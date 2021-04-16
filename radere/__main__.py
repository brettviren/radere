#!/usr/bin/env python3
'''
CLI to radere
'''

import click
import radere

@click.group()
@click.pass_context
def cli(ctx):
    '''
    radere command line interface
    '''
    ctx.obj = dict()

@cli.command()
def version():
    '''
    Print the version
    '''
    click.echo(radere.__version__)



def main():
    cli(obj=None)

if '__main__' == __name__:
    main()
