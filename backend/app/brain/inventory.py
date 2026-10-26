"""
Stage 5: SQLite-Backed Restaurant Table Inventory & Conflict Resolution Engine
for Haveli Grill & Courtyard (F-7 Markaz, Islamabad).

Provides:
- Realistic table layout across 3 seating zones (Indoor AC, Rooftop, Private Family Majlis)
- Fake seeded bookings creating realistic peak-hour conflicts (Today/Tomorrow 8:00 PM Indoor AC for 6+ guests)
- Best-fit table allocation (smallest available table with capacity >= party_size)
- Nearest-time (±30m, ±60m, ±90m) and alternative-seating recommender when full
- Full CRUD for creating, looking up, modifying, and cancelling reservations
"""

from __future__ import annotations

import datetime as dt
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.brain.slot_extractor import get_reference_date
from app.models import (
    AlternativeOption,
    BookingRecord,
    DEFAULT_SEATING,
    InventoryCheckResult,
    SeatingArea,
)


# All 30-minute reservation slots from 12:00 PM to 11:30 PM
ALL_TIME_SLOTS: List[str] = [
    "12:00", "12:30",
    "13:00", "13:30",
    "14:00", "14:30",
    "15:00", "15:30",
    "16:00", "16:30",
    "17:00", "17:30",
    "18:00", "18:30",
    "19:00", "19:30",
    "20:00", "20:30",
    "21:00", "21:30",
    "22:00", "22:30",
    "23:00", "23:30",
]

# Restaurant physical tables (Fake data)
SEED_TABLES: List[Tuple[str, str, str, int]] = [
    # (table_id, table_name, seating_preference, capacity)
    ("IND-T1", "Margalla Booth", SeatingArea.INDOOR_AC.value, 2),
    ("IND-T2", "Courtyard Window 4-Top", SeatingArea.INDOOR_AC.value, 4),
    ("IND-T3", "Chinar Indoor 5-Top", SeatingArea.INDOOR_AC.value, 5),
    ("IND-T4", "Khyber Family Table", SeatingArea.INDOOR_AC.value, 6),
    ("IND-T5", "Grand Dastarkhwan Table", SeatingArea.INDOOR_AC.value, 8),
    ("ROOF-T1", "Margalla Edge 2-Top", SeatingArea.ROOFTOP.value, 2),
    ("ROOF-T2", "Pergola Terrace 4-Top", SeatingArea.ROOFTOP.value, 4),
    ("ROOF-T3", "Charcoal Grill Terrace 6-Top", SeatingArea.ROOFTOP.value, 6),
    ("ROOF-T4", "Starlight Lounge 8-Top", SeatingArea.ROOFTOP.value, 8),
    ("ROOF-T5", "Chinar Pavilion 10-Top", SeatingArea.ROOFTOP.value, 10),
    ("MAJ-T1", "Clifton Majlis Room", SeatingArea.PRIVATE_MAJLIS.value, 8),
    ("MAJ-T2", "Shahi Darbar Suite", SeatingArea.PRIVATE_MAJLIS.value, 12),
    ("MAJ-T3", "Grand Haveli Hall", SeatingArea.PRIVATE_MAJLIS.value, 20),
]


def format_display_time(time_24h: str) -> str:
    """Converts '20:00' -> '8:00 PM', '12:30' -> '12:30 PM'."""
    try:
        hr_s, mn_s = time_24h.split(":")
        hr = int(hr_s)
        mn = int(mn_s)
        mer = "AM" if hr < 12 else "PM"
        hr_12 = hr % 12
        if hr_12 == 0:
            hr_12 = 12
        return f"{hr_12}:{mn:02d} {mer}"
    except Exception:
        return time_24h


def _snap_to_nearest_slot(time_24h: str) -> str:
    """Snaps arbitrary HH:MM (like 19:15) to the nearest 30-minute slot for table allocation."""
    if time_24h in ALL_TIME_SLOTS:
        return time_24h
    try:
        hr, mn = [int(x) for x in time_24h.split(":")]
        total = hr * 60 + mn
        best_slot = "20:00"
        best_diff = 10**9
        for s in ALL_TIME_SLOTS:
            sh, sm = [int(x) for x in s.split(":")]
            diff = abs((sh * 60 + sm) - total)
            if diff < best_diff:
                best_diff = diff
                best_slot = s
        return best_slot
    except Exception:
        return "20:00"


class InventoryManager:
    """SQLite-backed table inventory and reservation store."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            base_dir = Path(__file__).resolve().parent.parent.parent / "data"
            base_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(base_dir / "dawatdesk.db")
        self.db_path = db_path
        self._init_and_seed()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_and_seed(self, reference_date: Optional[str] = None) -> None:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS restaurant_tables (
                    table_id TEXT PRIMARY KEY,
                    table_name TEXT NOT NULL,
                    seating_preference TEXT NOT NULL,
                    capacity INTEGER NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS bookings (
                    booking_ref TEXT PRIMARY KEY,
                    customer_name TEXT NOT NULL,
                    phone_number TEXT NOT NULL,
                    party_size INTEGER NOT NULL,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    seating_preference TEXT NOT NULL,
                    table_id TEXT NOT NULL,
                    table_name TEXT NOT NULL,
                    special_requests TEXT,
                    status TEXT NOT NULL DEFAULT 'CONFIRMED',
                    created_at TEXT NOT NULL
                )
                """
            )
            cur.execute("SELECT COUNT(*) AS cnt FROM restaurant_tables")
            if cur.fetchone()["cnt"] == 0:
                cur.executemany(
                    "INSERT INTO restaurant_tables (table_id, table_name, seating_preference, capacity) VALUES (?, ?, ?, ?)",
                    SEED_TABLES,
                )

            cur.execute("SELECT COUNT(*) AS cnt FROM bookings")
            if cur.fetchone()["cnt"] == 0:
                self._insert_seed_bookings(cur, reference_date)

            conn.commit()
        finally:
            conn.close()

    def _insert_seed_bookings(self, cur: sqlite3.Cursor, reference_date: Optional[str] = None) -> None:
        ref = get_reference_date(reference_date)
        today_str = ref.isoformat()
        tomorrow_str = (ref + dt.timedelta(days=1)).isoformat()
        now_iso = dt.datetime.now(dt.timezone.utc).isoformat()

        # Peak-hour seeded bookings (all fake data):
        # Makes Today & Tomorrow at 20:00 (8:00 PM) Indoor AC FULL for 6+ guests (IND-T4 & IND-T5 booked).
        seed_bookings = [
            (
                "DD-1001",
                "Tariq Mehmood",
                "0300-1112233",
                6,
                today_str,
                "20:00",
                SeatingArea.INDOOR_AC.value,
                "IND-T4",
                "Khyber Family Table",
                "Window preference",
                "CONFIRMED",
                now_iso,
            ),
            (
                "DD-1003",
                "Nadia Hussain",
                "0321-4445566",
                8,
                today_str,
                "20:00",
                SeatingArea.INDOOR_AC.value,
                "IND-T5",
                "Grand Dastarkhwan Table",
                "Birthday table setup",
                "CONFIRMED",
                now_iso,
            ),
            (
                "DD-1002",
                "Kamran Akmal",
                "0333-2223344",
                6,
                tomorrow_str,
                "20:00",
                SeatingArea.INDOOR_AC.value,
                "IND-T4",
                "Khyber Family Table",
                None,
                "CONFIRMED",
                now_iso,
            ),
            (
                "DD-1004",
                "Fariha Pervez",
                "0345-6667788",
                8,
                tomorrow_str,
                "20:00",
                SeatingArea.INDOOR_AC.value,
                "IND-T5",
                "Grand Dastarkhwan Table",
                "High chair for child",
                "CONFIRMED",
                now_iso,
            ),
            (
                "DD-2045",
                "Salman Ahmad",
                "0301-9998877",
                4,
                tomorrow_str,
                "19:00",
                SeatingArea.ROOFTOP.value,
                "ROOF-T2",
                "Pergola Terrace 4-Top",
                "Anniversary table setup",
                "CONFIRMED",
                now_iso,
            ),
        ]
        cur.executemany(
            """
            INSERT OR REPLACE INTO bookings (
                booking_ref, customer_name, phone_number, party_size,
                date, time, seating_preference, table_id, table_name,
                special_requests, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            seed_bookings,
        )

    def reset_database(self, reference_date: Optional[str] = None) -> None:
        """Clears all bookings and re-inserts the clean seed data."""
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM bookings")
            cur.execute("DELETE FROM restaurant_tables")
            cur.executemany(
                "INSERT INTO restaurant_tables (table_id, table_name, seating_preference, capacity) VALUES (?, ?, ?, ?)",
                SEED_TABLES,
            )
            self._insert_seed_bookings(cur, reference_date)
            conn.commit()
        finally:
            conn.close()

    def _find_free_table(
        self,
        cur: sqlite3.Cursor,
        date_str: str,
        slot_time: str,
        party_size: int,
        seating_preference: str,
        exclude_booking_ref: Optional[str] = None,
    ) -> Optional[sqlite3.Row]:
        """
        Finds the smallest-capacity available table in `seating_preference`
        that can seat `party_size` at `(date_str, slot_time)`.
        """
        if exclude_booking_ref:
            cur.execute(
                """
                SELECT t.table_id, t.table_name, t.seating_preference, t.capacity
                FROM restaurant_tables t
                WHERE t.seating_preference = ?
                  AND t.capacity >= ?
                  AND t.table_id NOT IN (
                      SELECT b.table_id FROM bookings b
                      WHERE b.date = ?
                        AND b.time = ?
                        AND b.status = 'CONFIRMED'
                        AND b.booking_ref != ?
                  )
                ORDER BY t.capacity ASC, t.table_id ASC
                LIMIT 1
                """,
                (seating_preference, party_size, date_str, slot_time, exclude_booking_ref),
            )
        else:
            cur.execute(
                """
                SELECT t.table_id, t.table_name, t.seating_preference, t.capacity
                FROM restaurant_tables t
                WHERE t.seating_preference = ?
                  AND t.capacity >= ?
                  AND t.table_id NOT IN (
                      SELECT b.table_id FROM bookings b
                      WHERE b.date = ?
                        AND b.time = ?
                        AND b.status = 'CONFIRMED'
                  )
                ORDER BY t.capacity ASC, t.table_id ASC
                LIMIT 1
                """,
                (seating_preference, party_size, date_str, slot_time),
            )
        return cur.fetchone()

    def check_availability(
        self,
        date_str: str,
        time_str: str,
        party_size: int,
        seating_preference: Optional[str] = None,
        exclude_booking_ref: Optional[str] = None,
    ) -> InventoryCheckResult:
        """
        Checks if a table is free for (date_str, time_str, party_size, seating_preference).
        If full, computes the 2 closest available time slots in the requested zone PLUS
        any alternative seating zones open at the requested time.
        """
        target_seating = seating_preference or DEFAULT_SEATING
        slot_time = _snap_to_nearest_slot(time_str)

        conn = self._connect()
        try:
            cur = conn.cursor()
            match = self._find_free_table(
                cur,
                date_str=date_str,
                slot_time=slot_time,
                party_size=party_size,
                seating_preference=target_seating,
                exclude_booking_ref=exclude_booking_ref,
            )

            if match:
                return InventoryCheckResult(
                    checked=True,
                    available=True,
                    allocated_table_id=match["table_id"],
                    allocated_table_name=match["table_name"],
                    allocated_seating=match["seating_preference"],
                    allocated_capacity=match["capacity"],
                )

            # Conflict detected! Build nearest-time and alternative-seating recommendations.
            alternatives: List[AlternativeOption] = []

            # 1. Nearest time slots in the SAME seating zone (sorted by time distance)
            req_h, req_m = [int(x) for x in slot_time.split(":")]
            req_mins = req_h * 60 + req_m

            candidate_slots = sorted(
                [s for s in ALL_TIME_SLOTS if s != slot_time],
                key=lambda s: (
                    abs((int(s.split(":")[0]) * 60 + int(s.split(":")[1])) - req_mins),
                    int(s.split(":")[0]) * 60 + int(s.split(":")[1]),
                ),
            )

            nearest_time_count = 0
            for cand_time in candidate_slots:
                if nearest_time_count >= 2:
                    break
                alt_table = self._find_free_table(
                    cur,
                    date_str=date_str,
                    slot_time=cand_time,
                    party_size=party_size,
                    seating_preference=target_seating,
                    exclude_booking_ref=exclude_booking_ref,
                )
                if alt_table:
                    alternatives.append(
                        AlternativeOption(
                            date=date_str,
                            time=cand_time,
                            display_time=format_display_time(cand_time),
                            seating_preference=target_seating,
                            table_id=alt_table["table_id"],
                            table_name=alt_table["table_name"],
                            capacity=alt_table["capacity"],
                            reason="nearest_time",
                        )
                    )
                    nearest_time_count += 1

            # 2. Alternative seating areas at the EXACT requested time
            all_zones = [
                SeatingArea.INDOOR_AC.value,
                SeatingArea.ROOFTOP.value,
                SeatingArea.PRIVATE_MAJLIS.value,
            ]
            for alt_zone in all_zones:
                if alt_zone == target_seating:
                    continue
                alt_zone_table = self._find_free_table(
                    cur,
                    date_str=date_str,
                    slot_time=slot_time,
                    party_size=party_size,
                    seating_preference=alt_zone,
                    exclude_booking_ref=exclude_booking_ref,
                )
                if alt_zone_table:
                    alternatives.append(
                        AlternativeOption(
                            date=date_str,
                            time=slot_time,
                            display_time=format_display_time(slot_time),
                            seating_preference=alt_zone,
                            table_id=alt_zone_table["table_id"],
                            table_name=alt_zone_table["table_name"],
                            capacity=alt_zone_table["capacity"],
                            reason="alternative_seating",
                        )
                    )

            # Determine specific conflict reason (zone capacity vs time slot booked)
            cur.execute(
                "SELECT MAX(capacity) AS max_cap FROM restaurant_tables WHERE seating_preference = ?",
                (target_seating,),
            )
            max_cap_row = cur.fetchone()
            zone_max_cap = max_cap_row["max_cap"] if max_cap_row and max_cap_row["max_cap"] else 0

            if party_size > zone_max_cap:
                reason = (
                    f"{target_seating} tables seat up to {zone_max_cap} guests per table "
                    f"(requested {party_size} guests)."
                )
            else:
                reason = (
                    f"All {target_seating} tables for {party_size} guests are booked at "
                    f"{format_display_time(slot_time)} on {date_str}."
                )

            return InventoryCheckResult(
                checked=True,
                available=False,
                conflict_reason=reason,
                alternatives=alternatives,
            )
        finally:
            conn.close()

    def create_booking(
        self,
        customer_name: str,
        phone_number: str,
        party_size: int,
        date_str: str,
        time_str: str,
        seating_preference: str,
        special_requests: Optional[str] = None,
    ) -> Tuple[Optional[BookingRecord], InventoryCheckResult]:
        """
        Verifies availability and atomically inserts a new confirmed booking.
        """
        slot_time = _snap_to_nearest_slot(time_str)
        check = self.check_availability(
            date_str=date_str,
            time_str=slot_time,
            party_size=party_size,
            seating_preference=seating_preference,
        )
        if not check.available or not check.allocated_table_id:
            return None, check

        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) AS cnt FROM bookings")
            seq = int(cur.fetchone()["cnt"]) + 3001
            booking_ref = f"DD-{seq}"
            while True:
                cur.execute("SELECT 1 FROM bookings WHERE booking_ref = ?", (booking_ref,))
                if not cur.fetchone():
                    break
                seq += 1
                booking_ref = f"DD-{seq}"

            now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
            cur.execute(
                """
                INSERT INTO bookings (
                    booking_ref, customer_name, phone_number, party_size,
                    date, time, seating_preference, table_id, table_name,
                    special_requests, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'CONFIRMED', ?)
                """,
                (
                    booking_ref,
                    customer_name,
                    phone_number,
                    party_size,
                    date_str,
                    slot_time,
                    seating_preference,
                    check.allocated_table_id,
                    check.allocated_table_name or check.allocated_table_id,
                    special_requests,
                    now_iso,
                ),
            )
            conn.commit()
            record = BookingRecord(
                booking_ref=booking_ref,
                customer_name=customer_name,
                phone_number=phone_number,
                party_size=party_size,
                date=date_str,
                time=slot_time,
                seating_preference=seating_preference,
                table_id=check.allocated_table_id,
                table_name=check.allocated_table_name or check.allocated_table_id,
                special_requests=special_requests,
                status="CONFIRMED",
                created_at=now_iso,
            )
            return record, check
        finally:
            conn.close()

    def find_booking(
        self,
        booking_ref: Optional[str] = None,
        phone_number: Optional[str] = None,
        customer_name: Optional[str] = None,
    ) -> Optional[BookingRecord]:
        """Looks up an active CONFIRMED booking by reference code, phone number, or name."""
        conn = self._connect()
        try:
            cur = conn.cursor()
            row = None
            if booking_ref:
                cur.execute(
                    "SELECT * FROM bookings WHERE UPPER(booking_ref) = UPPER(?) AND status = 'CONFIRMED'",
                    (booking_ref,),
                )
                row = cur.fetchone()
            if not row and phone_number:
                cur.execute(
                    "SELECT * FROM bookings WHERE phone_number = ? AND status = 'CONFIRMED' ORDER BY created_at DESC LIMIT 1",
                    (phone_number,),
                )
                row = cur.fetchone()
            if not row and customer_name:
                cur.execute(
                    "SELECT * FROM bookings WHERE LOWER(customer_name) LIKE ? AND status = 'CONFIRMED' ORDER BY created_at DESC LIMIT 1",
                    (f"%{customer_name.lower()}%",),
                )
                row = cur.fetchone()

            if not row:
                return None

            return BookingRecord(**dict(row))
        finally:
            conn.close()

    def update_booking(
        self,
        booking_ref: str,
        party_size: Optional[int] = None,
        date_str: Optional[str] = None,
        time_str: Optional[str] = None,
        seating_preference: Optional[str] = None,
        special_requests: Optional[str] = None,
    ) -> Tuple[Optional[BookingRecord], InventoryCheckResult]:
        """Updates an existing booking after validating availability for the new parameters."""
        existing = self.find_booking(booking_ref=booking_ref)
        if not existing:
            return None, InventoryCheckResult(
                checked=True,
                available=False,
                conflict_reason=f"Booking {booking_ref} not found.",
            )

        new_party = party_size if party_size is not None else existing.party_size
        new_date = date_str if date_str is not None else existing.date
        new_time = _snap_to_nearest_slot(time_str) if time_str is not None else existing.time
        new_seating = seating_preference if seating_preference is not None else existing.seating_preference
        new_special = special_requests if special_requests is not None else existing.special_requests

        check = self.check_availability(
            date_str=new_date,
            time_str=new_time,
            party_size=new_party,
            seating_preference=new_seating,
            exclude_booking_ref=booking_ref,
        )
        if not check.available or not check.allocated_table_id:
            return None, check

        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE bookings
                SET party_size = ?, date = ?, time = ?, seating_preference = ?,
                    table_id = ?, table_name = ?, special_requests = ?
                WHERE UPPER(booking_ref) = UPPER(?)
                """,
                (
                    new_party,
                    new_date,
                    new_time,
                    new_seating,
                    check.allocated_table_id,
                    check.allocated_table_name or check.allocated_table_id,
                    new_special,
                    booking_ref,
                ),
            )
            conn.commit()
            updated = self.find_booking(booking_ref=booking_ref)
            return updated, check
        finally:
            conn.close()

    def cancel_booking(self, booking_ref: str) -> Optional[BookingRecord]:
        """Marks an existing booking as CANCELLED and frees its table."""
        existing = self.find_booking(booking_ref=booking_ref)
        if not existing:
            return None
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE bookings SET status = 'CANCELLED' WHERE UPPER(booking_ref) = UPPER(?)",
                (booking_ref,),
            )
            conn.commit()
            existing.status = "CANCELLED"
            return existing
        finally:
            conn.close()

    def list_bookings(self, include_cancelled: bool = False) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            cur = conn.cursor()
            if include_cancelled:
                cur.execute("SELECT * FROM bookings ORDER BY date ASC, time ASC")
            else:
                cur.execute(
                    "SELECT * FROM bookings WHERE status = 'CONFIRMED' ORDER BY date ASC, time ASC"
                )
            return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()

    def list_tables(self) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM restaurant_tables ORDER BY seating_preference, capacity")
            return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()
