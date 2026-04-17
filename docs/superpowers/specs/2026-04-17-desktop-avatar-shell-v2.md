# Design: Desktop Avatar Shell V2

**Date:** 2026-04-17  
**Author:** Khalil + Codex  
**Status:** Draft

## Overview

The desktop companion should behave like a modern, intelligent desktop pet:

- present
- useful
- emotionally readable
- non-invasive

The previous mockup direction was too panel-heavy. The shell must stop feeling like a floating web app and start feeling like a living companion layer over the desktop.

## Core Decision

Default behavior:

- show mostly the avatar head
- keep the control surface minimal
- make system state visibly alive

On intentional interaction:

- clicking the companion expands to reveal more of the body
- a reduce action collapses back to compact mode

This gives us a cleaner always-on experience without losing the value of a full-body anime companion.

## UX Modes

### 1. Compact

Default mode.

Characteristics:

- avatar is mostly head-first
- smallest persistent footprint
- horizontal action rail only
- status strip remains visible
- no detached settings panel

Purpose:

- companion stays present without blocking the screen
- strongest desktop-pet feeling

### 2. Expanded

Triggered by clicking the companion.

Characteristics:

- reveal more of the upper body or near full body
- same action rail stays attached
- a small amount of contextual conversation UI may appear
- still overlay-like, not a full app surface

Purpose:

- reward interaction
- increase presence during active conversation

### 3. Settings

Triggered from the gear button in the action rail.

Characteristics:

- settings open from the shell itself
- grounded visually near the companion
- small and focused
- should feel like companion configuration, not an admin dashboard

Purpose:

- configure behavior without breaking the avatar-first model

## Persistent Controls

The control rail should contain only the minimum useful actions:

- microphone
- interrupt / stop current response
- screen share / vision entry point
- settings

Notes:

- `interrupt` is preferred over a generic pause icon because the action is to stop current playback or generation immediately
- screen share can exist in the UI before full backend support, but must be visually honest if it is still preview-only

## Status System

The shell must feel alive even when the avatar is compact.

Required visible states:

- `Idle`
- `Listening`
- `Thinking`
- `Speaking`
- `Loading`
- `Error`
- `Disconnected`

Rules:

- state should be readable in one glance
- state should affect accent/glow subtly
- the UI must never hide failure states
- the shell should not pretend to be available when the backend is degraded

## Visual Direction

- transparent overlay first
- avatar is always the hero
- control chrome should feel like small floating islands
- cold blue-gray glass surface
- thin borders
- subtle cyan highlights
- no heavy dashboard cards
- no large settings rail floating in space

## Interaction Rules

- compact is the default entry point
- expansion is intentional, not automatic
- reduction must be easy and always available
- if auto-collapse exists later, it should only happen after inactivity and never during active speech

## Why This Direction

Showing full body all the time matches some anime desktop software, but it is too expensive in screen space for a companion expected to stay present continuously.

Showing only the head by default:

- improves coexistence with real work
- makes the assistant feel more like a desktop pet
- creates a stronger contrast between idle and engaged states

Expansion on click preserves the emotional value of the full character without making the shell intrusive.

## Mockup Reference

Primary browser mockup:

- [docs/mockups/desktop-avatar-shell-v1.html](/mnt/c/Users/Khalil/Documents/Local-AI-Companion/docs/mockups/desktop-avatar-shell-v1.html:1)

Live2D crop reference:

- [docs/mockups/desktop-avatar-live2d-preview.html](/mnt/c/Users/Khalil/Documents/Local-AI-Companion/docs/mockups/desktop-avatar-live2d-preview.html:1)
