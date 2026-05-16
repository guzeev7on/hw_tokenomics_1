import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

np.random.seed(42)

plt.rcParams['figure.figsize'] = (12, 4)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


class AMMPool:
    def __init__(self, x_token: float, y_stable: float):
        self.x = x_token
        self.y = y_stable

    @property
    def k(self) -> float:
        return self.x * self.y

    @property
    def price(self) -> float:
        return self.y / self.x if self.x > 0 else 0.0

    def swap_x_for_y(self, dx: float) -> float:
        assert dx > 0 and dx < self.x * 0.95, 'Слишком крупный swap'
        k = self.k
        self.x += dx
        new_y = k / self.x
        dy = self.y - new_y
        self.y = new_y
        return dy

    def swap_y_for_x(self, dy: float) -> float:
        assert dy > 0 and dy < self.y * 0.95, 'Слишком крупный swap'
        k = self.k
        self.y += dy
        new_x = k / self.y
        dx = self.x - new_x
        self.x = new_x
        return dx

    def add_liquidity(self, dx: float, dy: float):
        self.x += dx
        self.y += dy

    def remove_liquidity(self, fraction: float):
        assert 0 < fraction < 1
        removed_x = self.x * fraction
        removed_y = self.y * fraction
        self.x -= removed_x
        self.y -= removed_y
        return removed_x, removed_y


class McCallumPolicy:
    def __init__(self, target_growth: float = 0.02, theta: float = 0.5,
                 epoch_length: int = 100):
        self.target_growth = target_growth
        self.theta = theta
        self.epoch_length = epoch_length
        self.prev_volume_growth = 0.0
        self.prev_velocity_change = 0.0

    def compute_emission(self, current_supply: float,
                         volume_history: list,
                         supply_history: list) -> float:
        ep = self.epoch_length

        if len(volume_history) < 2 * ep:
            return current_supply * self.target_growth

        recent_vol = sum(volume_history[-ep:])
        prev_vol = sum(volume_history[-2 * ep:-ep])

        recent_supply = np.mean(supply_history[-ep:])
        prev_supply = np.mean(supply_history[-2 * ep:-ep])

        v_now = recent_vol / recent_supply if recent_supply > 0 else 1.0
        v_prev = prev_vol / prev_supply if prev_supply > 0 else 1.0

        delta_v = (v_now - v_prev) / v_prev if v_prev > 0 else 0.0
        volume_growth = (recent_vol - prev_vol) / prev_vol if prev_vol > 0 else 0.0

        delta_m = (
            self.target_growth
            - delta_v
            + self.theta * (self.target_growth - volume_growth)
        )

        delta_m = np.clip(delta_m, -0.10, 0.10)

        self.prev_volume_growth = volume_growth
        self.prev_velocity_change = delta_v

        return current_supply * delta_m


class FixedPolicy:
    def __init__(self, growth: float = 0.005):
        self.growth = growth

    def compute_emission(self, current_supply: float,
                         volume_history: list,
                         supply_history: list) -> float:
        return current_supply * self.growth


class QuadraticBondingPolicy:
    def __init__(self, target_price: float = 1.0, strength: float = 0.10):
        self.target_price = target_price
        self.strength = strength

    def compute_emission(self, current_supply: float, price: float) -> float:
        error = price / self.target_price - 1

        delta_m = self.strength * np.sign(error) * (error ** 2)
        delta_m = np.clip(delta_m, -0.10, 0.10)

        return current_supply * delta_m


class Agent:
    def __init__(self, name: str, tokens: float, stables: float):
        self.name = name
        self.tokens = tokens
        self.stables = stables

    def act(self, pool: AMMPool, price_history: list, step: int):
        raise NotImplementedError


class MomentumAgent(Agent):
    def __init__(self, name: str, tokens: float, stables: float,
                 window: int = 20, trade_frac: float = 0.03):
        super().__init__(name, tokens, stables)
        self.window = window
        self.trade_frac = trade_frac

    def act(self, pool: AMMPool, price_history: list, step: int) -> float:
        if len(price_history) < self.window:
            return 0.0

        ma = np.mean(price_history[-self.window:])
        current = pool.price
        volume = 0.0

        if current > ma:
            dy = min(self.stables * self.trade_frac, pool.y * 0.04)
            if dy > 0.01:
                dx = pool.swap_y_for_x(dy)
                self.stables -= dy
                self.tokens += dx
                volume = dy
        else:
            dx = min(self.tokens * self.trade_frac, pool.x * 0.04)
            if dx > 0.01:
                dy = pool.swap_x_for_y(dx)
                self.tokens -= dx
                self.stables += dy
                volume = dy

        return volume


class MeanReversionAgent(Agent):
    def __init__(self, name: str, tokens: float, stables: float,
                 window: int = 30, threshold: float = 0.05,
                 trade_frac: float = 0.04):
        super().__init__(name, tokens, stables)
        self.window = window
        self.threshold = threshold
        self.trade_frac = trade_frac

    def act(self, pool: AMMPool, price_history: list, step: int) -> float:
        if len(price_history) < self.window:
            return 0.0

        ma = np.mean(price_history[-self.window:])
        current = pool.price
        volume = 0.0

        if current < ma * (1 - self.threshold):
            dy = min(self.stables * self.trade_frac, pool.y * 0.04)
            if dy > 0.01:
                dx = pool.swap_y_for_x(dy)
                self.stables -= dy
                self.tokens += dx
                volume = dy

        elif current > ma * (1 + self.threshold):
            dx = min(self.tokens * self.trade_frac, pool.x * 0.04)
            if dx > 0.01:
                dy = pool.swap_x_for_y(dx)
                self.tokens -= dx
                self.stables += dy
                volume = dy

        return volume


class RandomAgent(Agent):
    def __init__(self, name: str, tokens: float, stables: float,
                 trade_frac: float = 0.02, activity: float = 0.3):
        super().__init__(name, tokens, stables)
        self.trade_frac = trade_frac
        self.activity = activity

    def act(self, pool: AMMPool, price_history: list, step: int) -> float:
        if np.random.random() > self.activity:
            return 0.0

        volume = 0.0
        size = np.random.uniform(0.005, self.trade_frac)

        if np.random.random() < 0.5:
            dy = min(self.stables * size, pool.y * 0.03)
            if dy > 0.01:
                dx = pool.swap_y_for_x(dy)
                self.stables -= dy
                self.tokens += dx
                volume = dy
        else:
            dx = min(self.tokens * size, pool.x * 0.03)
            if dx > 0.01:
                dy = pool.swap_x_for_y(dx)
                self.tokens -= dx
                self.stables += dy
                volume = dy

        return volume


class ShockGenerator:
    def __init__(self, demand_steps: list, supply_steps: list,
                 liquidity_steps: list):
        self.demand_steps = set(demand_steps)
        self.supply_steps = set(supply_steps)
        self.liquidity_steps = set(liquidity_steps)
        self.shock_log = []

    def apply(self, step: int, pool: AMMPool,
              current_supply: float) -> tuple:
        if step in self.demand_steps:
            dy = pool.y * 0.20
            pool.swap_y_for_x(dy)
            self.shock_log.append((step, 'demand', dy))
            return 'demand', dy

        if step in self.supply_steps:
            new_tokens = current_supply * 0.50
            pool.x += new_tokens
            self.shock_log.append((step, 'supply', new_tokens))
            return 'supply', new_tokens

        if step in self.liquidity_steps:
            rx, ry = pool.remove_liquidity(0.30)
            self.shock_log.append((step, 'liquidity', 0.30))
            return 'liquidity', rx + ry

        return None, 0.0


N_STEPS = 10000
EPOCH = 100
INIT_TOKEN = 10000
INIT_STABLE = 10000


def make_agents():
    return [
        MomentumAgent('Momentum_1', tokens=500, stables=500, window=20),
        MomentumAgent('Momentum_2', tokens=300, stables=700, window=50),
        MeanReversionAgent('MeanRev_1', tokens=600, stables=600, window=30),
        MeanReversionAgent('MeanRev_2', tokens=400, stables=800, window=15, threshold=0.03),
        RandomAgent('Random_1', tokens=500, stables=500, activity=0.4),
        RandomAgent('Random_2', tokens=500, stables=500, activity=0.2),
    ]


def make_shocks():
    return ShockGenerator(
        demand_steps=[2000, 5500, 8000],
        supply_steps=[3500, 7000],
        liquidity_steps=[4500, 9000],
    )


def run_simulation(policy_name: str):
    np.random.seed(42)

    pool = AMMPool(INIT_TOKEN, INIT_STABLE)
    agents = make_agents()
    shocks = make_shocks()

    if policy_name == 'mccallum':
        policy = McCallumPolicy(target_growth=0.02, theta=0.5, epoch_length=EPOCH)
    elif policy_name == 'fixed':
        policy = FixedPolicy(growth=0.005)
    elif policy_name == 'bonding':
        policy = QuadraticBondingPolicy(target_price=1.0, strength=0.10)
    else:
        raise ValueError('Неизвестное правило')

    history = {
        'price': [],
        'volume': [],
        'supply_x': [],
        'supply_y': [],
        'total_supply': [],
        'shock_type': [],
    }

    total_supply = INIT_TOKEN + sum(a.tokens for a in agents)

    for step in range(N_STEPS):
        step_volume = 0.0

        for agent in agents:
            try:
                vol = agent.act(pool, history['price'], step)
                step_volume += vol
            except AssertionError:
                pass

        shock_type, shock_vol = shocks.apply(step, pool, total_supply)
        step_volume += shock_vol

        if step > 0 and step % EPOCH == 0:
            if policy_name == 'bonding':
                emission = policy.compute_emission(total_supply, pool.price)
            else:
                emission = policy.compute_emission(
                    total_supply,
                    history['volume'],
                    history['total_supply']
                )

            if policy_name == 'bonding':
                if emission > 0:
                    pool.x += emission
                    total_supply += emission
                elif emission < 0:
                    burn = min(abs(emission), pool.x * 0.05)
                    pool.x -= burn
                    total_supply -= burn
            else:
                if emission > 0:
                    pool.add_liquidity(
                        emission * 0.5,
                        emission * 0.5 * pool.price
                    )
                    total_supply += emission
                elif emission < 0:
                    burn = min(abs(emission), pool.x * 0.05)
                    pool.x -= burn
                    total_supply -= burn

        history['price'].append(pool.price)
        history['volume'].append(step_volume)
        history['supply_x'].append(pool.x)
        history['supply_y'].append(pool.y)
        history['total_supply'].append(total_supply)
        history['shock_type'].append(shock_type)

    df = pd.DataFrame(history)
    df.index.name = 'step'

    return df, shocks


def plot_main(df: pd.DataFrame, title: str):
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

    ax = axes[0]
    ax.plot(df['price'], linewidth=0.7, color='black')
    for _, row in df[df['shock_type'].notna()].iterrows():
        color = {
            'demand': 'blue',
            'supply': 'red',
            'liquidity': 'orange'
        }[row['shock_type']]
        ax.axvline(row.name, color=color, alpha=0.5,
                   linewidth=1, linestyle='--')
    ax.set_ylabel('Цена токена')
    ax.set_title(f'ДИНАМИКА ЦЕНЫ — {title}')

    ax = axes[1]
    vol_ma = df['volume'].rolling(50).mean()
    ax.plot(vol_ma, linewidth=0.7, color='black')
    ax.set_ylabel('Объём MA50')
    ax.set_title('ОБЪЁМ ТОРГОВ')

    ax = axes[2]
    rolling_vol = df['volume'].rolling(100).sum()
    rolling_supply = df['total_supply'].rolling(100).mean()
    velocity = rolling_vol / rolling_supply
    ax.plot(velocity, linewidth=0.7, color='black')
    ax.set_ylabel('V')
    ax.set_title('СКОРОСТЬ ОБРАЩЕНИЯ')

    ax = axes[3]
    ax.plot(df['total_supply'], linewidth=0.7, color='black')
    ax.set_ylabel('Total Supply')
    ax.set_title('ПРЕДЛОЖЕНИЕ ТОКЕНА')
    ax.set_xlabel('Шаг симуляции')

    plt.tight_layout()
    plt.show()


def compute_garch(df: pd.DataFrame):
    log_returns = np.log(df['price'] / df['price'].shift(1)).dropna()
    log_returns = log_returns.replace([np.inf, -np.inf], np.nan).dropna()

    log_returns_scaled = log_returns * 100

    garch = arch_model(log_returns_scaled, vol='GARCH', p=1, q=1, mean='Zero')
    result = garch.fit(disp='off')

    return log_returns, log_returns_scaled, result


def plot_garch(log_returns_scaled: pd.Series, result, shocks, title: str):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(log_returns_scaled.values, linewidth=0.3, color='black')
    axes[0].set_ylabel('Log returns (%)')
    axes[0].set_title(f'ЛОГАРИФМИЧЕСКИЕ ПРИРАЩЕНИЯ — {title}')

    axes[1].plot(result.conditional_volatility, linewidth=0.7, color='black')
    axes[1].set_ylabel('σ_t')
    axes[1].set_title('УСЛОВНАЯ ВОЛАТИЛЬНОСТЬ GARCH')

    for s, stype, _ in shocks.shock_log:
        color = {
            'demand': 'blue',
            'supply': 'red',
            'liquidity': 'orange'
        }[stype]
        axes[1].axvline(s, color=color, alpha=0.4,
                        linewidth=1, linestyle='--')

    axes[1].set_xlabel('Шаг')
    plt.tight_layout()
    plt.show()


def compute_irf(df: pd.DataFrame, shock_steps: list,
                column: str = 'price',
                pre_window: int = 100,
                post_window: int = 500) -> pd.DataFrame:
    responses = []

    for step in shock_steps:
        if step - pre_window < 0 or step + post_window >= len(df):
            continue

        baseline = df[column].iloc[step - pre_window:step].mean()

        post = df[column].iloc[step:step + post_window].values
        deviation = (post - baseline) / baseline * 100

        responses.append(deviation)

    if not responses:
        return pd.DataFrame()

    responses = np.array(responses)

    result = pd.DataFrame({
        'mean': np.mean(responses, axis=0),
        'upper': np.percentile(responses, 84, axis=0),
        'lower': np.percentile(responses, 16, axis=0),
    })

    result.index.name = 'steps_after_shock'

    return result


def plot_irf(df: pd.DataFrame, shocks, title: str):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    shock_types = [
        ('demand', list(shocks.demand_steps), 'ШОК СПРОСА'),
        ('supply', list(shocks.supply_steps), 'ШОК ПРЕДЛОЖЕНИЯ'),
        ('liquidity', list(shocks.liquidity_steps), 'ШОК ЛИКВИДНОСТИ'),
    ]

    for ax, (stype, steps, shock_title) in zip(axes, shock_types):
        irf = compute_irf(df, steps, post_window=400)

        if len(irf) > 0:
            ax.plot(irf['mean'], color='black', linewidth=1.2)
            ax.fill_between(
                irf.index,
                irf['lower'],
                irf['upper'],
                alpha=0.15,
                color='gray'
            )

        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_title(shock_title)
        ax.set_xlabel('Шагов после шока')
        ax.set_ylabel('Отклонение цены (%)')

    plt.suptitle(f'ИМПУЛЬСНЫЕ ОТКЛИКИ — {title}', y=1.02)
    plt.tight_layout()
    plt.show()


def compute_metrics(df: pd.DataFrame, shocks, rule_name: str):
    log_returns, log_returns_scaled, result = compute_garch(df)

    price = df['price']

    std_price = log_returns.std() * 100

    cummax = price.cummax()
    drawdown = (price - cummax) / cummax * 100
    max_drawdown = drawdown.min()

    recovery_times = []

    shock_steps = (
        list(shocks.demand_steps)
        + list(shocks.supply_steps)
        + list(shocks.liquidity_steps)
    )

    for step in shock_steps:
        if step - 100 < 0 or step + 500 >= len(df):
            continue

        baseline = df['price'].iloc[step - 100:step].mean()

        for t in range(1, 500):
            diff = abs(df['price'].iloc[step + t] - baseline) / baseline
            if diff < 0.01:
                recovery_times.append(t)
                break
        else:
            recovery_times.append(500)

    avg_recovery = np.mean(recovery_times) if recovery_times else np.nan

    alpha = result.params['alpha[1]']
    beta = result.params['beta[1]']
    persistence = alpha + beta

    metrics = {
        'Правило': rule_name,
        'Волатильность': std_price,
        'Макс. просадка': max_drawdown,
        'Среднее время возврата': avg_recovery,
        'GARCH alpha': alpha,
        'GARCH beta': beta,
        'GARCH persistence': persistence,
    }

    return metrics, log_returns_scaled, result


def print_metrics(metrics: dict):
    print('=' * 70)
    print(metrics['Правило'])
    print('=' * 70)
    print(f'Волатильность: {metrics["Волатильность"]:.4f}')
    print(f'Макс. просадка: {metrics["Макс. просадка"]:.2f}%')
    print(f'Среднее время возврата: {metrics["Среднее время возврата"]:.0f}')
    print(f'GARCH alpha: {metrics["GARCH alpha"]:.4f}')
    print(f'GARCH beta: {metrics["GARCH beta"]:.4f}')
    print(f'GARCH persistence: {metrics["GARCH persistence"]:.4f}')
    print('=' * 70)


def main():
    cases = {
        'mccallum': 'Правило МакКаллума',
        'fixed': 'Фиксированное правило',
        'bonding': 'Quadratic bonding curve',
    }

    all_metrics = []

    for policy_name, title in cases.items():
        df, shocks = run_simulation(policy_name)

        print()
        print(f'Сценарий: {title}')
        print(f'Финальная цена: {df["price"].iloc[-1]:.4f}')
        print(f'Шоков произошло: {df["shock_type"].notna().sum()}')

        metrics, log_returns_scaled, result = compute_metrics(df, shocks, title)
        all_metrics.append(metrics)

        print_metrics(metrics)

        plot_main(df, title)
        plot_garch(log_returns_scaled, result, shocks, title)
        plot_irf(df, shocks, title)

    result_table = pd.DataFrame(all_metrics)

    print()
    print('=' * 90)
    print('СРАВНИТЕЛЬНАЯ ТАБЛИЦА')
    print('=' * 90)
    print(result_table.to_string(index=False))
    print('=' * 90)

    result_table.to_csv('comparison_metrics.csv', index=False)
    print('Таблица сохранена в файл: comparison_metrics.csv')


if __name__ == '__main__':
    main()
