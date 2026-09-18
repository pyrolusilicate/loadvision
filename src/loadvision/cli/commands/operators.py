"""CLI commands for operator accounts; the operator pages cannot create or change them."""

import argparse
import logging
from getpass import getpass

from loadvision.application.operators import (
    AddOperator,
    DisableOperator,
    ListOperators,
    SetOperatorPassword,
)
from loadvision.cli.database import unit_of_work_factory
from loadvision.cli.errors import CommandError
from loadvision.domain.timestamps import format_utc_seconds, utc_now

_logger = logging.getLogger(__name__)


def add_commands(groups: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    """Adds the `operators` group and its commands.

    Args:
        groups: Subparsers of the top-level parser.
    """
    parser = groups.add_parser("operators", help="accounts of the operator pages")
    commands = parser.add_subparsers(title="commands", required=True)

    add = commands.add_parser(
        "add",
        help="create an account",
        description="Creates an account; the password is asked for twice and never echoed.",
    )
    add.add_argument("login", help="lowercase login; it also signs the labels this person sets")
    add.set_defaults(run=add_operator)

    password = commands.add_parser(
        "password",
        help="set a new password",
        description="Sets a new password, enables the account and ends its open sessions.",
    )
    password.add_argument("login", help="account login")
    password.set_defaults(run=set_password)

    disable = commands.add_parser(
        "disable",
        help="stop an account from signing in",
        description="Disables the account and ends its open sessions at once.",
    )
    disable.add_argument("login", help="account login")
    disable.set_defaults(run=disable_operator)

    listing = commands.add_parser("list", help="accounts", description="Accounts, by login.")
    listing.set_defaults(run=list_operators)


def add_operator(args: argparse.Namespace) -> int:
    """Creates an account.

    Returns:
        Exit code 0.

    Raises:
        CommandError: If the login is malformed or taken, or the passwords are too short or
            differ.
        RuntimeError: If `DATABASE_URL` is not set.
    """
    password = _new_password()
    with unit_of_work_factory() as uow_factory:
        try:
            operator = AddOperator(uow_factory=uow_factory, clock=utc_now)(args.login, password)
        except ValueError as error:
            raise CommandError(str(error)) from error
    _logger.info("account %s created", operator.login)
    return 0


def set_password(args: argparse.Namespace) -> int:
    """Sets a new password.

    Returns:
        Exit code 0.

    Raises:
        CommandError: If there is no such account, or the passwords are too short or differ.
        RuntimeError: If `DATABASE_URL` is not set.
    """
    password = _new_password()
    with unit_of_work_factory() as uow_factory:
        try:
            SetOperatorPassword(uow_factory=uow_factory)(args.login, password)
        except ValueError as error:
            raise CommandError(str(error)) from error
    _logger.info("password of %s changed; its sessions are closed", args.login)
    return 0


def disable_operator(args: argparse.Namespace) -> int:
    """Disables an account.

    Returns:
        Exit code 0.

    Raises:
        CommandError: If there is no such account.
        RuntimeError: If `DATABASE_URL` is not set.
    """
    with unit_of_work_factory() as uow_factory:
        try:
            DisableOperator(uow_factory=uow_factory)(args.login)
        except ValueError as error:
            raise CommandError(str(error)) from error
    _logger.info("account %s disabled; its sessions are closed", args.login)
    return 0


def list_operators(_args: argparse.Namespace) -> int:
    """Prints the accounts.

    Returns:
        Exit code 0.

    Raises:
        RuntimeError: If `DATABASE_URL` is not set.
    """
    with unit_of_work_factory() as uow_factory:
        operators = ListOperators(uow_factory=uow_factory)()
    if not operators:
        _logger.info("no accounts yet; create one with: loadvision operators add <login>")
        return 0
    _logger.info("%-24s%-10s%s", "login", "active", "created")
    for operator in operators:
        _logger.info(
            "%-24s%-10s%s",
            operator.login,
            "yes" if operator.active else "no",
            format_utc_seconds(operator.created_at),
        )
    return 0


def _new_password() -> str:
    password = getpass("password: ")
    if getpass("repeat it: ") != password:
        raise CommandError("the passwords differ")
    return password
