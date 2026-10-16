import * as pty from 'node-pty';
import { execFileSync } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { stripVTControlCharacters } from 'node:util';
import { setTimeout as sleep } from 'node:timers/promises';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const output = resolve(root, 'repro773', 'transcripts');
const executable = execFileSync('where.exe', ['julia'], { encoding: 'utf8' }).trim().split(/\r?\n/)[0];
const julia = execFileSync(executable, ['--version'], { encoding: 'utf8' }).trim();
const fit = (extra, marker) => 'import SymbolicRegression: SRRegressor, machine, fit!; '
  + 'X = (a = rand(500), b = rand(500)); '
  + 'y = @. 2 * cos(X.a * 23.5) - X.b ^ 2; '
  + 'y = y .+ randn(500) .* 1e-3; '
  + `model = SRRegressor(niterations=3, binary_operators=[+,-,*], unary_operators=[cos]${extra}); `
  + `fit!(machine(model,X,y)); :${marker}`;
const scenarios = [
  { name: 'A_control', line: ':noop_done', marker: 'noop_done' },
  { name: 'B_base_only', line: 'Base.start_reading(stdin); sleep(2); Base.stop_reading(stdin); :b_done', marker: 'b_done' },
  { name: 'C_sr_reader', line: 'using SymbolicRegression; let S = SymbolicRegression.SearchUtilsModule; r = S.watch_stream(stdin); sleep(2); S.check_for_user_quit(r); S.close_reader!(r); end; :c_done', marker: 'c_done' },
  { name: 'D_readme_fit', line: fit('', 'd_done'), marker: 'd_done', timeout: 600_000 },
  { name: 'E_readme_devnull', line: fit(', input_stream=devnull', 'e_done'), marker: 'e_done', timeout: 600_000 },
];
mkdirSync(output, { recursive: true });
console.log(JSON.stringify({ julia, platform: process.platform, node: process.version, terminal: 'node-pty ConPTY', cols: 120, rows: 40 }));

async function run(scenario) {
  let terminal;
  let raw = '';
  let exited = false;
  let result = 'HARNESS_FAIL';
  let detail = '';
  const events = [];
  const event = (kind, data = {}) => events.push({ time: new Date().toISOString(), offset: raw.length, kind, ...data });
  const plainSince = (offset) => stripVTControlCharacters(raw.slice(offset).replace(/\x1b\[\d+;1[Hf]/g, '\n'));
  const prompt = (text) => /julia>\s*$/.test(text);
  const value = (text, expected) => new RegExp(`(?:^|[\\r\\n])\\s*${expected}\\s*(?:[\\r\\n])`).test(text);
  async function waitFor(predicate, offset, timeout, phase) {
    const deadline = Date.now() + timeout;
    while (Date.now() < deadline) {
      if (predicate(plainSince(offset))) {
        event('matched', { phase });
        return true;
      }
      if (exited) throw new Error(`Julia exited during ${phase}`);
      await sleep(25);
    }
    event('timeout', { phase });
    return false;
  }
  async function type(line) {
    event('keyboard', { text: `${line}\r` });
    for (const character of line) {
      terminal.write(character);
      await sleep(2);
    }
    terminal.write('\r');
  }
  try {
    terminal = pty.spawn(executable, [`--project=${root}`, '--startup-file=no', '--color=no'], {
      name: 'xterm-256color', cols: 120, rows: 40, cwd: root,
      env: { ...process.env, TERM: 'xterm-256color' }, useConpty: true,
    });
    terminal.onData((data) => { raw += data; });
    terminal.onExit((status) => { exited = true; event('exit', status); });
    event('spawn', { pid: terminal.pid });
    if (!await waitFor(prompt, 0, 120_000, 'initial prompt')) throw new Error('Initial prompt never appeared');
    let offset = raw.length;
    await type('1+1');
    if (!await waitFor((text) => value(text, '2') && prompt(text), offset, 30_000, 'sanity 1+1')) {
      throw new Error('Sanity 1+1 did not print 2 and return to prompt');
    }
    offset = raw.length;
    await type(scenario.line);
    if (!await waitFor((text) => value(text, `:${scenario.marker}`) && prompt(text), offset, scenario.timeout ?? 180_000, 'trigger')) {
      throw new Error('Trigger did not print its marker and return to prompt');
    }
    offset = raw.length;
    await type('3+4');
    result = await waitFor((text) => value(text, '7') && prompt(text), offset, 30_000, 'post-trigger 3+4') ? 'PASS' : 'HANG';
    event('result', { result, postTriggerOffset: offset });
  } catch (error) {
    detail = error.stack ?? String(error);
    event('error', { detail });
  } finally {
    if (terminal && !exited) {
      event('kill');
      terminal.kill();
      await sleep(500);
    }
    writeFileSync(resolve(output, `${scenario.name}.raw.txt`), raw);
    writeFileSync(resolve(output, `${scenario.name}.events.json`), JSON.stringify(events, null, 2));
  }
  console.log(`${scenario.name}: ${result}${detail ? `\n${detail}` : ''}`);
  return { scenario: scenario.name, julia, result, detail };
}

const results = [];
for (const scenario of scenarios) results.push(await run(scenario));
console.log('scenario | julia | result');
for (const row of results) console.log(`${row.scenario} | ${row.julia} | ${row.result}`);
writeFileSync(resolve(output, 'results.json'), JSON.stringify(results, null, 2));
