import test from 'node:test'; import assert from 'node:assert/strict'; import { slugify } from '../src/slug.mjs';
test('trims dashes', () => assert.equal(slugify('  Hello World! '), 'hello-world'));
test('polish letters', () => assert.equal(slugify('Zażółć gęślą jaźń'), 'zazolc-gesla-jazn'));
test('collapses', () => assert.equal(slugify('a---b'), 'a-b'));
