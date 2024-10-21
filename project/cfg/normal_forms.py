from pyformlang.cfg import CFG, Production, Epsilon, Variable


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    cnf_cfg = cfg.to_normal_form()
    productions = set(cnf_cfg.productions).union(
        Production(nullable.value, [Epsilon()])
        for nullable in cfg.get_nullable_symbols()
    )

    return CFG(
        start_symbol=cnf_cfg.start_symbol,
        productions=productions,
    ).remove_useless_symbols()
